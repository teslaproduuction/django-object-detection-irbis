import os
import io
import numpy as np
from PIL import Image as I
import torch
import collections
from ast import literal_eval
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

from django.views.generic.detail import DetailView
from django.views.generic import FormView
from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib import messages
from django.conf import settings
from django.shortcuts import render, get_object_or_404, redirect
from django.core.paginator import Paginator
from django.urls import reverse
from django.utils.translation import gettext_lazy as _
from django.http import JsonResponse, HttpResponseRedirect
from django.db.models import Q

from images.models import ImageFile
from .models import InferencedImage, ManualAnnotation
from .forms import InferencedImageForm, YoloModelForm, ManualAnnotationFormSetFactory
from modelmanager.models import MLModel


def calculate_iou(box1, box2):
    """
    Вычисляет IoU (Intersection over Union) между двумя bounding box'ами.
    Формат box: {'x': center_x, 'y': center_y, 'width': width, 'height': height}
    """

    # Преобразуем из центра в углы
    def center_to_corners(box):
        x_center, y_center, width, height = box['x'], box['y'], box['width'], box['height']
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        return x1, y1, x2, y2

    x1_1, y1_1, x2_1, y2_1 = center_to_corners(box1)
    x1_2, y1_2, x2_2, y2_2 = center_to_corners(box2)

    # Вычисляем область пересечения
    x1_inter = max(x1_1, x1_2)
    y1_inter = max(y1_1, y1_2)
    x2_inter = min(x2_1, x2_2)
    y2_inter = min(y2_1, y2_2)

    # Если нет пересечения
    if x1_inter >= x2_inter or y1_inter >= y2_inter:
        return 0.0

    # Площадь пересечения
    intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)

    # Площади боксов
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)

    # Площадь объединения
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def filter_automatic_detections_by_manual(automatic_detections, manual_detections, iou_threshold=0.1):
    """
    Фильтрует автоматические детекции, удаляя те, которые пересекаются с ручными аннотациями.

    Args:
        automatic_detections: список автоматических детекций
        manual_detections: список ручных аннотаций
        iou_threshold: порог IoU для считания пересечения (по умолчанию 0.1)

    Returns:
        список автоматических детекций без пересечений с ручными
    """
    if not manual_detections:
        return automatic_detections

    filtered_detections = []

    for auto_detection in automatic_detections:
        # Проверяем, пересекается ли автоматическая детекция с любой ручной аннотацией
        has_overlap = False

        for manual_detection in manual_detections:
            iou = calculate_iou(auto_detection, manual_detection)
            print(
                f"Проверяем пересечение: авто '{auto_detection.get('class', 'unknown')}' vs ручная '{manual_detection.get('class', 'unknown')}', IoU={iou:.3f}")

            if iou > iou_threshold:
                has_overlap = True
                print(f"✅ УДАЛЯЕМ автоматическую детекцию класса '{auto_detection.get('class', 'unknown')}' "
                      f"из-за пересечения (IoU={iou:.3f}) с ручной аннотацией класса '{manual_detection.get('class', 'unknown')}'")
                break

        # Если нет пересечения, добавляем автоматическую детекцию
        if not has_overlap:
            print(
                f"✅ СОХРАНЯЕМ автоматическую детекцию класса '{auto_detection.get('class', 'unknown')}' - нет пересечений")
            filtered_detections.append(auto_detection)

    return filtered_detections


def combine_detections_with_priority(automatic_detections, manual_detections, iou_threshold=0.1):
    """
    Объединяет автоматические и ручные детекции с приоритетом ручных.

    Args:
        automatic_detections: список автоматических детекций
        manual_detections: список ручных аннотаций
        iou_threshold: порог IoU для считания пересечения

    Returns:
        объединенный список детекций с приоритетом ручных аннотаций
    """
    print(f"🔄 ОБЪЕДИНЕНИЕ ДЕТЕКЦИЙ:")
    print(f"  📊 Автоматических детекций: {len(automatic_detections)}")
    print(f"  ✋ Ручных аннотаций: {len(manual_detections)}")

    # Фильтруем автоматические детекции, убирая пересечения с ручными
    filtered_auto = filter_automatic_detections_by_manual(
        automatic_detections, manual_detections, iou_threshold
    )

    # Объединяем: сначала ручные (приоритет), потом отфильтрованные автоматические
    combined = manual_detections + filtered_auto

    print(f"  ✅ Автоматических после фильтрации: {len(filtered_auto)}")
    print(f"  🎯 Итого объединенных детекций: {len(combined)}")

    return combined


class SaveAnnotationView(LoginRequiredMixin, FormView):
    """API для сохранения аннотаций через AJAX запросы."""

    def post(self, request, *args, **kwargs):
        if not request.is_ajax():
            return JsonResponse({'status': 'error', 'message': 'Only AJAX requests are allowed'}, status=400)

        image_id = request.POST.get('image_id')
        class_name = request.POST.get('class_name')
        x_center = float(request.POST.get('x_center', 0))
        y_center = float(request.POST.get('y_center', 0))
        width = float(request.POST.get('width', 0))
        height = float(request.POST.get('height', 0))

        if not image_id or not class_name:
            return JsonResponse({'status': 'error', 'message': 'Missing required data'}, status=400)

        try:
            image = ImageFile.objects.get(id=image_id)
            annotation = ManualAnnotation.objects.create(
                image=image,
                class_name=class_name,
                x_center=x_center,
                y_center=y_center,
                width=width,
                height=height,
                created_by=request.user,
                is_manual=True
            )

            # Применяем аннотации к дубликатам
            self._apply_annotations_to_duplicates(image, annotation)

            # Обновляем существующие инференс-записи с учетом приоритета ручных аннотаций
            self._update_existing_inferences(image)

            return JsonResponse({
                'status': 'success',
                'message': _('Аннотация успешно сохранена'),
                'annotation_id': annotation.id
            })
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

    def _apply_annotations_to_duplicates(self, original_image, annotation):
        """Применяет аннотацию к изображениям-дубликатам."""
        if original_image.image_hash:
            duplicate_images = ImageFile.objects.filter(
                image_hash=original_image.image_hash
            ).exclude(id=original_image.id)

            for dup_img in duplicate_images:
                # Создаем копию аннотации для дубликата
                ManualAnnotation.objects.create(
                    image=dup_img,
                    class_name=annotation.class_name,
                    x_center=annotation.x_center,
                    y_center=annotation.y_center,
                    width=annotation.width,
                    height=annotation.height,
                    confidence=annotation.confidence,
                    created_by=annotation.created_by,
                    is_manual=True
                )

    def _update_existing_inferences(self, image):
        """Обновляет существующие инференс-записи с учетом приоритета ручных аннотаций."""
        manual_annotations = ManualAnnotation.objects.filter(image=image)
        manual_detection_info = [annotation.to_detection_format() for annotation in manual_annotations]

        existing_inferences = InferencedImage.objects.filter(orig_image=image)
        for inf_img in existing_inferences:
            is_custom_model = inf_img.custom_model is not None

            if inf_img.detection_info and is_custom_model:
                # Для кастомных моделей объединяем с приоритетом ручных аннотаций
                combined_info = combine_detections_with_priority(
                    inf_img.detection_info, manual_detection_info
                )
                inf_img.detection_info = combined_info
                inf_img.save()
                print(f"✅ SaveAnnotation: Обновлена кастомная модель с приоритетом: {len(combined_info)} детекций")
            elif inf_img.detection_info and not is_custom_model:
                # Для стандартных YOLO моделей НЕ обновляем
                print(f"🚫 SaveAnnotation: Стандартная YOLO модель - ручные аннотации НЕ применяются")


class ManualAnnotationView(LoginRequiredMixin, FormView):
    """Представление для ручной разметки изображений."""

    template_name = 'detectobj/manual_annotation.html'
    form_class = ManualAnnotationFormSetFactory

    def get_success_url(self):
        return reverse('detectobj:detection_image_detail_url', kwargs={'pk': self.image.id})

    def dispatch(self, request, *args, **kwargs):
        # Получаем изображение из URL
        self.image = get_object_or_404(ImageFile, id=self.kwargs['pk'])
        return super().dispatch(request, *args, **kwargs)

    def get_form_kwargs(self):
        kwargs = super().get_form_kwargs()
        kwargs['image'] = self.image
        kwargs['user'] = self.request.user
        return kwargs

    def get_initial(self):
        # Загружаем существующие аннотации для редактирования
        initial = []

        # Получаем аннотации для текущего изображения
        for annotation in ManualAnnotation.objects.filter(image=self.image):
            initial.append({
                'class_name': annotation.class_name,
                'x_center': annotation.x_center,
                'y_center': annotation.y_center,
                'width': annotation.width,
                'height': annotation.height,
            })

        # Если нет аннотаций для текущего изображения, ищем в дубликатах
        if not initial and self.image.image_hash:
            duplicate_images = ImageFile.objects.filter(
                image_hash=self.image.image_hash
            ).exclude(id=self.image.id)

            for dup_img in duplicate_images:
                dup_annotations = ManualAnnotation.objects.filter(image=dup_img)
                if dup_annotations.exists():
                    messages.info(
                        self.request,
                        _("Найдены аннотации с идентичного изображения из другого набора. Они загружены для редактирования.")
                    )
                    for annotation in dup_annotations:
                        initial.append({
                            'class_name': annotation.class_name,
                            'x_center': annotation.x_center,
                            'y_center': annotation.y_center,
                            'width': annotation.width,
                            'height': annotation.height,
                        })
                    break  # Берем аннотации только от первого найденного дубликата

        return initial

    def form_valid(self, form):
        # Save the manual annotations
        form.save()
        messages.success(self.request, _("Аннотации успешно сохранены"))

        # Применяем аннотации к дубликатам
        self._apply_annotations_to_duplicates()

        # Теперь обновляем существующие InferencedImage записи с учетом приоритета ручных аннотаций
        all_annotations = ManualAnnotation.objects.filter(image=self.image)

        if all_annotations.exists():
            # Преобразуем все ручные аннотации в формат детекции
            manual_detection_info = [annotation.to_detection_format() for annotation in all_annotations]

            # Обновляем записи инференса ТОЛЬКО для кастомных моделей
            existing_inferences = InferencedImage.objects.filter(orig_image=self.image)
            updated_count = 0

            for inf_img in existing_inferences:
                is_custom_model = inf_img.custom_model is not None

                if is_custom_model:
                    # Для кастомных моделей применяем приоритет ручных аннотаций
                    # Получаем оригинальные детекции модели (без старых ручных аннотаций)
                    original_detections = inf_img.detection_info or []

                    # Фильтруем только автоматические детекции (убираем старые ручные)
                    automatic_detections = []
                    for item in original_detections:
                        # Определяем автоматические детекции по отсутствию точного совпадения с ручными
                        is_automatic = True
                        for manual_item in manual_detection_info:
                            if (abs(item.get('x', 0) - manual_item.get('x', 0)) < 0.01 and
                                    abs(item.get('y', 0) - manual_item.get('y', 0)) < 0.01 and
                                    abs(item.get('width', 0) - manual_item.get('width', 0)) < 0.01 and
                                    abs(item.get('height', 0) - manual_item.get('height', 0)) < 0.01):
                                is_automatic = False
                                break

                        if is_automatic:
                            automatic_detections.append(item)

                    # Объединяем с приоритетом ручных аннотаций
                    combined_info = combine_detections_with_priority(
                        automatic_detections, manual_detection_info
                    )

                    # Обновляем detection_info
                    inf_img.detection_info = combined_info

                    # Перегенерируем изображение с аннотациями
                    try:
                        self._regenerate_inference_image(inf_img, combined_info)
                    except Exception as e:
                        print(f"Ошибка при перегенерации изображения инференса: {e}")

                    # Сохраняем запись
                    inf_img.save()
                    updated_count += 1

                    print(
                        f"✅ Кастомная модель: обновлена с приоритетом ({len(automatic_detections)} авто + {len(manual_detection_info)} ручных = {len(combined_info)})")
                else:
                    # Для стандартных YOLO моделей НЕ применяем ручные аннотации
                    print(f"🚫 Стандартная YOLO модель: ручные аннотации НЕ применяются к существующему инференсу")

            if updated_count > 0:
                messages.info(self.request,
                              _("Аннотации обновлены в %d записях с кастомными моделями") % updated_count)

        # Make sure to always redirect to the success URL
        return HttpResponseRedirect(self.get_success_url())

    def _regenerate_inference_image(self, inf_img, combined_detections):
        """Перегенерирует изображение инференса с обновленными аннотациями."""
        # Получаем путь к изображению инференса
        inf_img_path = inf_img.inf_image_path
        if inf_img_path.startswith(settings.MEDIA_URL):
            inf_img_path = inf_img_path[len(settings.MEDIA_URL):]
        inf_img_full_path = os.path.join(settings.MEDIA_ROOT, inf_img_path)

        # Удаляем старый файл если существует
        if os.path.exists(inf_img_full_path):
            os.remove(inf_img_full_path)
            print(f"🗑️ Удален старый файл: {inf_img_full_path}")

        # Открываем оригинальное изображение
        img_path = self.image.get_imagepath
        img = I.open(img_path)

        # Разделяем детекции на автоматические и ручные
        manual_detections = [d for d in combined_detections if d.get('is_manual', False)]
        automatic_detections = [d for d in combined_detections if not d.get('is_manual', False)]

        if inf_img.custom_model:
            # Для кастомной модели
            if manual_detections:
                # Если есть ручные аннотации, рендерим ТОЛЬКО их на чистом изображении
                annotator = Annotator(np.array(img))  # Используем оригинальное изображение

                # Добавляем только ручные аннотации красным цветом
                for detection in manual_detections:
                    # Преобразуем нормализованные координаты в пиксельные
                    img_width, img_height = img.size
                    x_center = detection.get('x', 0)
                    y_center = detection.get('y', 0)
                    width = detection.get('width', 0)
                    height = detection.get('height', 0)

                    x1 = int((x_center - width / 2) * img_width)
                    y1 = int((y_center - height / 2) * img_height)
                    x2 = int((x_center + width / 2) * img_width)
                    y2 = int((y_center + height / 2) * img_height)

                    # Красный цвет для ручных аннотаций
                    color = (255, 0, 0)
                    annotator.box_label([x1, y1, x2, y2], detection.get('class', 'unknown'), color=color)

                # Сохраняем изображение ТОЛЬКО с ручными аннотациями
                I.fromarray(annotator.result()).save(inf_img_full_path, format="JPEG")
                print(f"✅ Перегенерация: кастомная модель с ТОЛЬКО ручными аннотациями")
            else:
                # Если нет ручных аннотаций, запускаем стандартный инференс
                model = YOLO(inf_img.custom_model.pth_filepath)
                results = model(img,
                                conf=float(inf_img.model_conf) if inf_img.model_conf else settings.MODEL_CONFIDENCE,
                                verbose=False)
                result = results[0]
                plotted_img = result.plot()
                I.fromarray(plotted_img).save(inf_img_full_path, format="JPEG")
                print(f"✅ Перегенерация: кастомная модель с автоматическими детекциями")

        else:
            # Для YOLO модели - стандартная логика (не должно вызываться, но на всякий случай)
            if inf_img.yolo_model:
                yolo_weightsdir = settings.YOLOV8_WEIGTHS_DIR
                model_path = os.path.join(yolo_weightsdir, inf_img.yolo_model)
                if os.path.exists(model_path):
                    model = YOLO(model_path)
                else:
                    model = YOLO(inf_img.yolo_model)

                results = model(img,
                                conf=float(inf_img.model_conf) if inf_img.model_conf else settings.MODEL_CONFIDENCE,
                                verbose=False)

                result = results[0]
                plotted_img = result.plot()
                I.fromarray(plotted_img).save(inf_img_full_path, format="JPEG")
                print(f"📊 Перегенерация: стандартная YOLO модель")

    def _apply_annotations_to_duplicates(self):
        """Применяет аннотации текущего изображения к его дубликатам."""
        if self.image.image_hash:
            duplicate_images = ImageFile.objects.filter(
                image_hash=self.image.image_hash
            ).exclude(id=self.image.id)

            if duplicate_images.exists():
                all_annotations = ManualAnnotation.objects.filter(image=self.image)

                for dup_img in duplicate_images:
                    # Удаляем старые аннотации дубликата
                    ManualAnnotation.objects.filter(image=dup_img).delete()

                    # Копируем новые аннотации
                    for annotation in all_annotations:
                        ManualAnnotation.objects.create(
                            image=dup_img,
                            class_name=annotation.class_name,
                            x_center=annotation.x_center,
                            y_center=annotation.y_center,
                            width=annotation.width,
                            height=annotation.height,
                            confidence=annotation.confidence,
                            created_by=self.request.user,
                            is_manual=True
                        )

                messages.info(
                    self.request,
                    _("Аннотации также применены к %d идентичным изображениям в других наборах") % duplicate_images.count()
                )

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['image'] = self.image

        # Получаем список уникальных классов из всех аннотаций
        class_names = ManualAnnotation.objects.values_list('class_name', flat=True).distinct()
        context['class_names'] = list(class_names)

        # Добавляем размеры изображения для правильной работы JavaScript
        img_width, img_height, *_ = self.image.get_imgshape()
        context['img_width'] = img_width
        context['img_height'] = img_height

        # Проверяем наличие дубликатов
        if self.image.image_hash:
            duplicate_count = ImageFile.objects.filter(
                image_hash=self.image.image_hash
            ).exclude(id=self.image.id).count()

            if duplicate_count > 0:
                context['has_duplicates'] = True
                context['duplicate_count'] = duplicate_count

        return context


class InferenceImageDetectionView(LoginRequiredMixin, DetailView):
    model = ImageFile
    template_name = "detectobj/select_inference_image.html"

    def _get_all_annotations_for_image(self, img_qs):
        """Получает все аннотации для изображения, включая аннотации с дубликатов."""
        # Сначала получаем аннотации для самого изображения
        annotations = list(ManualAnnotation.objects.filter(image=img_qs))

        # Если у изображения есть хэш, ищем аннотации в дубликатах
        if img_qs.image_hash:
            duplicate_images = ImageFile.objects.filter(
                image_hash=img_qs.image_hash
            ).exclude(id=img_qs.id)

            for dup_img in duplicate_images:
                dup_annotations = ManualAnnotation.objects.filter(image=dup_img)
                if dup_annotations.exists():
                    # Добавляем аннотации из дубликатов
                    annotations.extend(list(dup_annotations))
                    break  # Берем аннотации только от первого найденного дубликата

        # Удаляем дубликаты аннотаций по координатам
        unique_annotations = []
        seen_coords = set()

        for annotation in annotations:
            coord_key = (
                round(annotation.x_center, 3),
                round(annotation.y_center, 3),
                round(annotation.width, 3),
                round(annotation.height, 3),
                annotation.class_name
            )
            if coord_key not in seen_coords:
                seen_coords.add(coord_key)
                unique_annotations.append(annotation)

        return unique_annotations

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        img_qs = self.get_object()
        imgset = img_qs.image_set
        images_qs = imgset.images.all()

        # For pagination GET request
        self.get_pagination(context, images_qs)

        # Получаем последнюю инференс-картинку
        if is_inf_img := InferencedImage.objects.filter(
                orig_image=img_qs
        ).exists():
            inf_img_qs = InferencedImage.objects.filter(orig_image=img_qs).first()
            context['inf_img_qs'] = inf_img_qs

            # Добавляем счетчик классов, если есть результаты обнаружения
            if inf_img_qs.detection_info:
                classes_list = [item.get('class') for item in inf_img_qs.detection_info]
                context['results_counter'] = collections.Counter(classes_list)

        # Получаем ВСЕ ручные аннотации (включая дубликаты) ТОЛЬКО для кастомных моделей
        manual_annotations = self._get_all_annotations_for_image(img_qs)

        # Фильтруем ручные аннотации в зависимости от типа модели
        if 'inf_img_qs' in context:
            inf_img = context['inf_img_qs']
            is_custom_inference = inf_img.custom_model is not None

            if not is_custom_inference:
                # Для стандартных YOLO моделей скрываем ручные аннотации
                manual_annotations = []
                print(f"🚫 Стандартная YOLO модель: ручные аннотации скрыты из контекста")

        context['manual_annotations'] = manual_annotations

        # Информация о дубликатах
        if img_qs.image_hash:
            duplicate_images = ImageFile.objects.filter(
                image_hash=img_qs.image_hash
            ).exclude(id=img_qs.id)

            if duplicate_images.exists():
                context['has_duplicates'] = True
                context['duplicate_count'] = duplicate_images.count()
                context['duplicate_sets'] = [dup.image_set.name for dup in duplicate_images[:3]]  # Первые 3 набора

        # Если есть инференс, добавляем к нему ручные аннотации для отображения
        if 'inf_img_qs' in context and manual_annotations:
            inf_img = context['inf_img_qs']
            detection_info = inf_img.detection_info or []

            # Проверяем, использовалась ли кастомная модель
            is_custom_inference = inf_img.custom_model is not None

            # Добавляем ручные аннотации к результатам детекции
            manual_detection_info = [annotation.to_detection_format() for annotation in manual_annotations]

            # Добавляем маркеры для различения типов детекций
            for item in detection_info:
                if 'is_manual' not in item:
                    item['is_manual'] = item.get('confidence', 0) == 1.0  # Предполагаем, что confidence=1.0 это ручная

            for item in manual_detection_info:
                item['is_manual'] = True

            print(f"🎯 КОНТЕКСТ: Применяем логику для отображения...")
            print(f"  📊 Детекций в inf_img: {len(detection_info)}")
            print(f"  ✋ Ручных аннотаций: {len(manual_detection_info)}")
            print(f"  🎯 Кастомная модель в инференсе: {is_custom_inference}")

            # Обновляем информацию о результатах в зависимости от типа модели
            if manual_detection_info:
                if is_custom_inference:
                    # Для кастомных моделей применяем приоритет ручных аннотаций
                    automatic_detections = [item for item in detection_info if not item.get('is_manual', False)]

                    combined_detection_info = combine_detections_with_priority(
                        automatic_detections, manual_detection_info
                    )
                    context['all_detection_info'] = combined_detection_info
                    print(f"✅ Кастомная модель: финальных детекций для отображения: {len(combined_detection_info)}")
                else:
                    # Для стандартных YOLO моделей просто объединяем без фильтрации
                    combined_detection_info = detection_info + manual_detection_info
                    context['all_detection_info'] = combined_detection_info
                    print(f"✅ Стандартная YOLO: объединяем без фильтрации: {len(combined_detection_info)}")

                # Обновляем счетчик классов для всех результатов
                all_classes = [item.get('class') for item in context['all_detection_info']]
                context['all_results_counter'] = collections.Counter(all_classes)
            else:
                context['all_detection_info'] = detection_info

        # Добавляем ссылку на страницу ручной разметки ТОЛЬКО для кастомных моделей
        context['manual_annotation_url'] = reverse('detectobj:manual_annotation_url', kwargs={'pk': img_qs.id})

        # Определяем, показывать ли кнопку ручной разметки
        show_manual_annotation = True
        if 'inf_img_qs' in context:
            inf_img = context['inf_img_qs']
            is_custom_inference = inf_img.custom_model is not None
            show_manual_annotation = is_custom_inference
            if not is_custom_inference:
                print(f"🚫 Стандартная YOLO модель: кнопка ручной разметки скрыта")

        context['show_manual_annotation'] = show_manual_annotation

        context["img_qs"] = img_qs
        context["form1"] = YoloModelForm()
        context["form2"] = InferencedImageForm()
        return context

    def get_pagination(self, context, images_qs):
        paginator = Paginator(
            images_qs, settings.PAGINATE_DETECTION_IMAGES_NUM)
        page_number = self.request.GET.get('page')
        page_obj = paginator.get_page(page_number)
        context["is_paginated"] = (
                images_qs.count() > settings.PAGINATE_DETECTION_IMAGES_NUM
        )
        context["page_obj"] = page_obj

    def post(self, request, *args, **kwargs):
        img_qs = self.get_object()
        img_bytes = img_qs.image.read()
        img = I.open(io.BytesIO(img_bytes))

        # Initialize inf_img_qs to None to avoid UnboundLocalError later
        inf_img_qs = None

        # Get form data
        modelconf = self.request.POST.get("model_conf")
        modelconf = float(
            modelconf) if modelconf else settings.MODEL_CONFIDENCE
        custom_model_id = self.request.POST.get("custom_model")
        yolo_model_name = self.request.POST.get("yolo_model")

        # Yolov8 dirs
        yolo_weightsdir = settings.YOLOV8_WEIGTHS_DIR

        # Flag to track if we're using a custom model
        is_custom_model = False

        # Whether user selected a custom model for the detection task
        # An offline model will be used for detection provided user has
        # uploaded this model.
        if custom_model_id:
            detection_model = MLModel.objects.get(id=custom_model_id)
            model = YOLO(detection_model.pth_filepath)
            is_custom_model = True
            # Print class names for debugging
            print("Custom model class names:", model.names)

        # Whether user selected a yolo model for the detection task
        # Selected yolov8 model will be downloaded, and ready for object
        # detection task. YOLOv8 API will start working.
        elif yolo_model_name:
            model_path = os.path.join(yolo_weightsdir, yolo_model_name)
            # Download if not exists
            if not os.path.exists(model_path):
                model = YOLO(yolo_model_name)  # This will download the model
            else:
                model = YOLO(model_path)
            # Print class names for debugging
            print("YOLO model class names:", model.names)

        # If using custom model, pre-update class names before inference
        if is_custom_model:
            # Fix class names in model before running inference
            for idx in model.names:
                if model.names[idx].lower() in ['cat', 'sheep']:
                    model.names[idx] = 'irbis'
                    print(f"Updated class {idx} from 'cat'/'sheep' to 'irbis'")

        # Run inference
        results = model(img, conf=modelconf, verbose=False)

        # Process results
        results_list = []
        for r in results:
            # Update class names in results if using custom model
            if is_custom_model and hasattr(r, 'names'):
                for idx in r.names:
                    if r.names[idx].lower() in ['cat', 'sheep']:
                        r.names[idx] = 'irbis'

            detection_boxes = []
            for box in r.boxes:
                # Convert box data to dictionary
                b = box.xywhn[0].tolist()  # normalized xywh format
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())

                # Get class name (will already be updated if custom model)
                class_name = model.names[cls_id]

                # Double check and update if still cat/sheep
                if is_custom_model and class_name.lower() in ['cat', 'giraffe']:
                    class_name = 'irbis'
                    model.names[cls_id] = 'irbis'

                detection_boxes.append({
                    "class": class_name,
                    "class_id": cls_id,
                    "confidence": conf,
                    "x": b[0],
                    "y": b[1],
                    "width": b[2],
                    "height": b[3]
                })
            results_list = detection_boxes

        # Get class occurrences
        classes_list = [item["class"] for item in results_list]
        results_counter = collections.Counter(classes_list)

        # Получаем ВСЕ ручные аннотации (включая дубликаты)
        manual_annotations = self._get_all_annotations_for_image(img_qs)

        # Create directory for inference images if it doesn't exist
        media_folder = settings.MEDIA_ROOT
        inferenced_img_dir = os.path.join(
            media_folder, "inferenced_image")
        if not os.path.exists(inferenced_img_dir):
            os.makedirs(inferenced_img_dir, exist_ok=True)

        # Set default values for variables used later
        img_filename = None
        inf_img = None

        if not results_list and not (manual_annotations and is_custom_model):
            messages.warning(
                request,
                _('Модель не смогла обнаружить объекты. Попробуйте другую модель или выполните ручную разметку.'))
        else:
            # Make sure class names are updated in results before plotting
            if is_custom_model:
                # Update class names in each result
                for r in results:
                    if hasattr(r, 'names'):
                        for idx in r.names:
                            if r.names[idx].lower() in ['cat', 'giraffe']:
                                r.names[idx] = 'irbis'

            # Save the annotated image with a unique name based on model type
            if custom_model_id:
                img_file_suffix = f"custom_{custom_model_id}"
            else:
                img_file_suffix = f"yolo_{os.path.splitext(yolo_model_name)[0]}"

            # Generate filename with model identifier to allow multiple detections of same image
            img_filename = f"{os.path.splitext(img_qs.name)[0]}_{img_file_suffix}{os.path.splitext(img_qs.name)[1]}"
            img_path = os.path.join(inferenced_img_dir, img_filename)

            # Create a new inference image record each time (don't use get_or_create)
            inf_img = InferencedImage(
                orig_image=img_qs,
                inf_image_path=f"{settings.MEDIA_URL}inferenced_image/{img_filename}",
            )

            # Обрабатываем ручные аннотации с приоритетом ТОЛЬКО для кастомных моделей
            manual_detection_info = []
            if manual_annotations and is_custom_model:  # Приоритет только для кастомных моделей
                manual_detection_info = [annotation.to_detection_format() for annotation in manual_annotations]
                # Добавляем специальный маркер для ручных аннотаций
                for item in manual_detection_info:
                    item['is_manual'] = True
                    item['confidence'] = 1.0  # Ручные аннотации всегда имеют confidence 1.0

            # Добавляем маркер для автоматических детекций
            for item in results_list:
                item['is_manual'] = False

            print(f"🔍 ОБРАБОТКА ДЕТЕКЦИЙ:")
            print(f"  📊 Автоматических детекций: {len(results_list)}")
            print(f"  ✋ Ручных аннотаций: {len(manual_annotations) if manual_annotations else 0}")
            print(f"  🎯 Кастомная модель: {is_custom_model}")

            # Объединяем детекции с приоритетом ручных аннотаций ТОЛЬКО для кастомных моделей
            if manual_detection_info and is_custom_model:
                print("🎯 Применяем приоритет ручных аннотаций для кастомной модели...")
                combined_detection_info = combine_detections_with_priority(
                    results_list, manual_detection_info
                )
                inf_img.detection_info = combined_detection_info
                print(f"✅ Финальное количество детекций: {len(combined_detection_info)}")
            elif manual_annotations and not is_custom_model:
                # Для стандартных YOLO моделей просто добавляем ручные аннотации БЕЗ фильтрации
                all_manual_detection_info = [annotation.to_detection_format() for annotation in manual_annotations]
                for item in all_manual_detection_info:
                    item['is_manual'] = True
                combined_info = results_list + all_manual_detection_info
                inf_img.detection_info = combined_info
                print(
                    f"📊 Стандартная YOLO модель: объединяем без фильтрации ({len(results_list)} авто + {len(all_manual_detection_info)} ручных = {len(combined_info)})")
            else:
                inf_img.detection_info = results_list
                print(f"📊 Только автоматические детекции: {len(results_list)}")

            inf_img.model_conf = modelconf
            if custom_model_id:
                inf_img.custom_model = detection_model
            elif yolo_model_name:
                inf_img.yolo_model = yolo_model_name

            # Создание изображения с аннотациями
            if manual_annotations and is_custom_model:
                # Для кастомной модели с ручными аннотациями рендерим ТОЛЬКО ручные аннотации на чистом изображении
                all_manual_detection_info = [annotation.to_detection_format() for annotation in manual_annotations]
                annotator = Annotator(np.array(img))  # ЧИСТОЕ изображение без автоматических детекций

                # Draw ONLY manual annotations in red
                for annotation in all_manual_detection_info:
                    # Get normalized coordinates
                    x_center = annotation.get('x', 0)
                    y_center = annotation.get('y', 0)
                    width = annotation.get('width', 0)
                    height = annotation.get('height', 0)

                    # Convert to pixel coordinates
                    img_width, img_height = img.size
                    x1 = int((x_center - width / 2) * img_width)
                    y1 = int((y_center - height / 2) * img_height)
                    x2 = int((x_center + width / 2) * img_width)
                    y2 = int((y_center + height / 2) * img_height)

                    # Set color (manual annotations in red)
                    color = (255, 0, 0)  # RGB for red

                    # Draw the box
                    annotator.box_label([x1, y1, x2, y2], annotation.get('class', 'unknown'), color=color)

                # Save the image with ONLY manual annotations
                I.fromarray(annotator.result()).save(img_path, format="JPEG")
                print(f"✅ Кастомная модель: сохранено ЧИСТОЕ изображение ТОЛЬКО с ручными аннотациями")

            elif results_list:
                # Для всех остальных случаев (стандартная YOLO или кастомная без ручных аннотаций)
                result = results[0]
                plotted_img = result.plot()
                I.fromarray(plotted_img).save(img_path, format="JPEG")

                if is_custom_model:
                    print(f"✅ Кастомная модель: сохранено изображение с автоматическими детекциями (нет ручных)")
                else:
                    print(f"📊 Стандартная YOLO: сохранено изображение с автоматическими детекциями")

            elif manual_annotations and is_custom_model:
                # No model results, just manual annotations (только для кастомных моделей)
                all_manual_detection_info = [annotation.to_detection_format() for annotation in manual_annotations]
                annotator = Annotator(np.array(img))

                # Draw manual annotations
                for annotation in all_manual_detection_info:
                    # Get normalized coordinates
                    x_center = annotation.get('x', 0)
                    y_center = annotation.get('y', 0)
                    width = annotation.get('width', 0)
                    height = annotation.get('height', 0)

                    # Convert to pixel coordinates
                    img_width, img_height = img.size
                    x1 = int((x_center - width / 2) * img_width)
                    y1 = int((y_center - height / 2) * img_height)
                    x2 = int((x_center + width / 2) * img_width)
                    y2 = int((y_center + height / 2) * img_height)

                    # Set color (manual annotations in red)
                    color = (255, 0, 0)  # RGB for red

                    # Draw the box
                    annotator.box_label([x1, y1, x2, y2], annotation.get('class', 'unknown'), color=color)

                # Save the image with manual annotations
                I.fromarray(annotator.result()).save(img_path, format="JPEG")
                print(f"✅ Кастомная модель: сохранено изображение только с ручными аннотациями")

            # Save inference image
            inf_img.save()

            # Get the latest inference for display
            if inf_img:
                inf_img_qs = inf_img

        # Clear CUDA cache if using GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # set image is_inferenced to true
        img_qs.is_inferenced = True
        img_qs.save()

        # Ready for rendering next image on same html page.
        imgset = img_qs.image_set
        images_qs = imgset.images.all()

        # For pagination POST request
        context = {}
        self.get_pagination(context, images_qs)

        context["img_qs"] = img_qs
        context["inferenced_img_dir"] = f"{settings.MEDIA_URL}inferenced_image/{img_filename}" if img_filename else None
        context["results_list"] = results_list
        context["results_counter"] = results_counter
        context["form1"] = YoloModelForm()
        context["form2"] = InferencedImageForm()

        # Add manual annotations to context
        context["manual_annotations"] = manual_annotations

        # Add link to manual annotation page ТОЛЬКО для кастомных моделей
        context['manual_annotation_url'] = reverse('detectobj:manual_annotation_url', kwargs={'pk': img_qs.id})

        # Определяем, показывать ли кнопку ручной разметки
        show_manual_annotation = is_custom_model  # В POST методе используем флаг из обработки
        context['show_manual_annotation'] = show_manual_annotation

        if not is_custom_model:
            print(f"🚫 POST: Стандартная YOLO модель - кнопка ручной разметки скрыта")

        # Add the latest inferenced image to the context if available
        if inf_img_qs:
            context["inf_img_qs"] = inf_img_qs

        return render(self.request, self.template_name, context)
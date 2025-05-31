import os
import io
from PIL import Image as I
import torch
from ultralytics import YOLO

from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.contrib.auth.mixins import LoginRequiredMixin
from django.views.generic.detail import DetailView
from django.views.generic.edit import CreateView, UpdateView, DeleteView
from django.views.generic.list import ListView
from django.views.generic import FormView, TemplateView
from django.urls import reverse_lazy, reverse
from django.http import Http404, JsonResponse
from django.db import models
from django.conf import settings
from django import forms
from django.utils.translation import gettext_lazy as _
from django.core.paginator import Paginator

from .models import ImageSet, ImageFile
from detectobj.models import InferencedImage
from modelmanager.models import MLModel


class ImageSetListView(LoginRequiredMixin, ListView):
    model = ImageSet
    template_name = "images/imageset_list.html"
    context_object_name = "imagesets"
    paginate_by = 12

    def get_queryset(self):
        view_type = self.request.GET.get('view')
        if view_type == 'public':
            return ImageSet.objects.filter(public=True).order_by('-created')
        else:
            return ImageSet.objects.filter(user=self.request.user).order_by('-created')

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        public_imagesets = ImageSet.objects.filter(
            public=True).order_by('-created')
        user_imagesets = ImageSet.objects.filter(
            user=self.request.user).order_by('-created')
        context["public_imagesets"] = public_imagesets
        context["user_imagesets"] = user_imagesets
        context["view_type"] = self.request.GET.get('view', 'personal')
        return context


class ImageSetDetailView(LoginRequiredMixin, DetailView):
    model = ImageSet
    template_name = "images/imageset_detail.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)

        # Статистика обработки
        context['processed_count'] = self.object.images.filter(is_inferenced=True).count()
        context['unprocessed_count'] = self.object.images.filter(is_inferenced=False).count()
        context['with_objects_count'] = self.object.images.filter(
            detectedimages__isnull=False
        ).distinct().count()

        return context


class ImageSetCreateView(LoginRequiredMixin, CreateView):
    model = ImageSet
    fields = ["name", "description", "public"]
    template_name = "images/imageset_form.html"

    def form_valid(self, form):
        form.instance.user = self.request.user
        return super().form_valid(form)


class ImageSetUpdateView(LoginRequiredMixin, UpdateView):
    model = ImageSet
    fields = ["name", "description", "public"]
    template_name = "images/imageset_form.html"

    def get_object(self, queryset=None):
        obj = super().get_object(queryset)
        if obj.user != self.request.user and not self.request.user.is_superuser:
            raise Http404("You can only edit your own imagesets")
        return obj


class ImageSetDeleteView(LoginRequiredMixin, DeleteView):
    """View для удаления набора изображений."""
    model = ImageSet
    template_name = 'images/imageset_confirm_delete.html'
    success_url = reverse_lazy('images:imageset_list_url')
    context_object_name = 'imageset'

    def get_object(self, queryset=None):
        """Ограничиваем удаление только для владельца набора."""
        obj = super().get_object(queryset)
        if obj.user != self.request.user and not self.request.user.is_superuser:
            raise Http404("Вы можете удалять только свои наборы изображений")
        return obj

    def delete(self, request, *args, **kwargs):
        """Переопределяем метод delete для добавления сообщения об успешном удалении."""
        imageset = self.get_object()
        imageset_name = imageset.name
        response = super().delete(request, *args, **kwargs)

        messages.success(
            request,
            f'Набор изображений "{imageset_name}" и все связанные изображения успешно удалены.'
        )
        return response


class ImageFileCreateView(LoginRequiredMixin, CreateView):
    model = ImageFile
    fields = ["image"]
    template_name = "images/imagefile_form.html"

    def dispatch(self, request, *args, **kwargs):
        self.imageset = get_object_or_404(ImageSet, id=self.kwargs['imageset_id'])
        return super().dispatch(request, *args, **kwargs)

    def form_valid(self, form):
        form.instance.image_set = self.imageset
        return super().form_valid(form)

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['imageset'] = self.imageset
        return context

    def get_success_url(self):
        return reverse('images:imageset_detail_url', kwargs={'pk': self.imageset.id})


class ImageFileListView(LoginRequiredMixin, ListView):
    model = ImageFile
    template_name = "images/imagefile_list.html"
    context_object_name = "images"
    paginate_by = 20

    def dispatch(self, request, *args, **kwargs):
        self.imageset = get_object_or_404(ImageSet, id=self.kwargs['imageset_id'])
        return super().dispatch(request, *args, **kwargs)

    def get_queryset(self):
        return self.imageset.images.all().order_by('-created')

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['imageset'] = self.imageset
        return context


class ImageFileDeleteView(LoginRequiredMixin, DeleteView):
    model = ImageFile
    template_name = "images/imagefile_confirm_delete.html"

    def get_object(self, queryset=None):
        obj = super().get_object(queryset)
        if obj.image_set.user != self.request.user and not self.request.user.is_superuser:
            raise Http404("You can only delete images from your own imagesets")
        return obj

    def get_success_url(self):
        return reverse('images:imagefile_list_url', kwargs={'imageset_id': self.object.image_set.id})


class BatchDetectionForm(forms.Form):
    """Форма для выбора модели для пакетного распознавания."""

    MODEL_TYPE_CHOICES = [
        ('yolo', _('Модель YOLO')),
        ('custom', _('Кастомная модель')),
    ]

    model_type = forms.ChoiceField(
        choices=MODEL_TYPE_CHOICES,
        widget=forms.RadioSelect,
        label=_('Тип модели'),
        initial='yolo'
    )

    yolo_model = forms.ChoiceField(
        choices=InferencedImage.YOLOMODEL_CHOICES,
        label=_('YOLO модель'),
        required=False,
        initial=InferencedImage.YOLOMODEL_CHOICES[0][0]
    )

    custom_model = forms.ModelChoiceField(
        queryset=MLModel.objects.none(),
        label=_('Кастомная модель'),
        required=False,
        empty_label=_('Выберите модель')
    )

    model_conf = forms.DecimalField(
        label=_("Порог уверенности модели"),
        max_value=1,
        min_value=0.25,
        max_digits=3,
        decimal_places=2,
        initial=0.45,
        help_text=_("Порог уверенности модели для предсказаний."),
    )

    def __init__(self, *args, **kwargs):
        user = kwargs.pop('user', None)
        super().__init__(*args, **kwargs)

        if user:
            # Показываем модели пользователя и публичные модели
            self.fields['custom_model'].queryset = MLModel.objects.filter(
                models.Q(uploader=user) | models.Q(public=True)
            ).order_by('name')


class BatchDetectionView(LoginRequiredMixin, FormView):
    """View для пакетного распознавания изображений в наборе."""

    template_name = 'images/batch_detection.html'
    form_class = BatchDetectionForm

    def dispatch(self, request, *args, **kwargs):
        self.imageset = get_object_or_404(ImageSet, id=self.kwargs['pk'])
        return super().dispatch(request, *args, **kwargs)

    def get_form_kwargs(self):
        kwargs = super().get_form_kwargs()
        kwargs['user'] = self.request.user
        return kwargs

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['imageset'] = self.imageset
        context['total_images'] = self.imageset.images.count()

        # Получаем статистику предыдущих обработок
        processed_images = InferencedImage.objects.filter(
            orig_image__image_set=self.imageset
        ).values('orig_image').distinct().count()

        context['processed_images'] = processed_images
        context['unprocessed_images'] = context['total_images'] - processed_images

        return context

    def form_valid(self, form):
        """Обработка формы и запуск пакетного распознавания."""
        model_type = form.cleaned_data['model_type']
        model_conf = float(form.cleaned_data['model_conf'])

        # Определяем какую модель использовать
        if model_type == 'yolo':
            yolo_model_name = form.cleaned_data['yolo_model']
            custom_model = None
        else:
            custom_model = form.cleaned_data['custom_model']
            yolo_model_name = None

        if not yolo_model_name and not custom_model:
            form.add_error('custom_model', _('Выберите модель для распознавания'))
            return self.form_invalid(form)

        # Запускаем пакетное распознавание
        results = self.process_batch_detection(
            yolo_model_name, custom_model, model_conf
        )

        # Сохраняем результаты в сессии для отображения
        self.request.session['batch_results'] = {
            'imageset_id': self.imageset.id,
            'processed_count': results['processed_count'],
            'detected_count': results['detected_count'],
            'detected_images': results['detected_images'],
            'model_type': model_type,
            'model_name': yolo_model_name or custom_model.name,
            'model_conf': model_conf,
        }

        return redirect('images:batch_detection_results_url', pk=self.imageset.id)

    def process_batch_detection(self, yolo_model_name, custom_model, model_conf):
        """Выполняет пакетное распознавание для всех изображений в наборе."""

        # Определяем тип модели
        is_custom_model = custom_model is not None

        # Загрузка модели
        if is_custom_model:
            # ИСПРАВЛЕНИЕ: используем pth_file вместо model_file
            model = YOLO(custom_model.pth_file.path)
            img_file_suffix = f"custom_{custom_model.id}"
        else:
            model = YOLO(yolo_model_name)  # Автоматически загружает модель по имени
            # Безопасно извлекаем имя модели без расширения
            model_name = yolo_model_name
            if '.' in model_name:
                model_name = model_name.split('.')[0]
            img_file_suffix = f"yolo_{model_name}"

        # Обновляем названия классов для кастомной модели
        if is_custom_model:
            for idx in model.names:
                if model.names[idx].lower() in ['cat', 'sheep']:
                    model.names[idx] = 'irbis'

        # Создаем директорию для сохранения результатов
        media_folder = settings.MEDIA_ROOT
        inferenced_img_dir = os.path.join(media_folder, "inferenced_image")
        if not os.path.exists(inferenced_img_dir):
            os.makedirs(inferenced_img_dir, exist_ok=True)

        processed_count = 0
        detected_count = 0
        detected_images = []

        # Обрабатываем каждое изображение в наборе
        for image_obj in self.imageset.images.all():
            try:
                # Загружаем изображение
                img_bytes = image_obj.image.read()
                img = I.open(io.BytesIO(img_bytes))

                # Запускаем распознавание
                results = model(img, conf=model_conf, verbose=False)

                # Обрабатываем результаты
                detection_boxes = []
                for r in results:
                    # Обновляем названия классов если нужно
                    if is_custom_model and hasattr(r, 'names'):
                        for idx in r.names:
                            if r.names[idx].lower() in ['cat', 'sheep']:
                                r.names[idx] = 'irbis'

                    for box in r.boxes:
                        b = box.xywhn[0].tolist()
                        cls_id = int(box.cls[0].item())
                        conf = float(box.conf[0].item())

                        class_name = model.names[cls_id]
                        if is_custom_model and class_name.lower() in ['cat', 'sheep']:
                            class_name = 'irbis'

                        detection_boxes.append({
                            "class": class_name,
                            "class_id": cls_id,
                            "confidence": conf,
                            "x": b[0],
                            "y": b[1],
                            "width": b[2],
                            "height": b[3]
                        })

                processed_count += 1

                # Если есть детекции, сохраняем результат
                if detection_boxes:
                    detected_count += 1

                    # Генерируем имя файла
                    if custom_model:
                        img_file_suffix = f"custom_{custom_model.id}"
                    else:
                        img_file_suffix = f"yolo_{os.path.splitext(yolo_model_name)[0]}"

                    img_filename = f"{os.path.splitext(image_obj.name)[0]}_{img_file_suffix}{os.path.splitext(image_obj.name)[1]}"
                    img_path = os.path.join(inferenced_img_dir, img_filename)

                    # Сохраняем изображение с аннотациями
                    result = results[0]
                    plotted_img = result.plot()
                    I.fromarray(plotted_img).save(img_path, format="JPEG")

                    # Создаем запись в базе данных
                    inf_img = InferencedImage.objects.create(
                        orig_image=image_obj,
                        inf_image_path=f"{settings.MEDIA_URL}inferenced_image/{img_filename}",
                        detection_info=detection_boxes,
                        model_conf=model_conf,
                        custom_model=custom_model if is_custom_model else None,
                        yolo_model=yolo_model_name if not is_custom_model else None,
                    )

                    # ИСПРАВЛЕНИЕ: используем get_imageurl без скобок (это свойство, а не метод)
                    detected_images.append({
                        'id': image_obj.id,
                        'name': image_obj.name,
                        'url': image_obj.get_imageurl,
                        'inference_url': inf_img.inf_image_path,
                        'detections_count': len(detection_boxes),
                        'classes': list(set([d['class'] for d in detection_boxes]))
                    })

                    # Помечаем изображение как обработанное
                    image_obj.is_inferenced = True
                    image_obj.save()

            except Exception as e:
                print(f"Ошибка при обработке изображения {image_obj.name}: {e}")
                continue

        # Очищаем кэш CUDA если доступен
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            'processed_count': processed_count,
            'detected_count': detected_count,
            'detected_images': detected_images,
        }


class BatchDetectionResultsView(LoginRequiredMixin, TemplateView):
    """View для отображения результатов пакетного распознавания."""

    template_name = 'images/batch_detection_results.html'

    def dispatch(self, request, *args, **kwargs):
        self.imageset = get_object_or_404(ImageSet, id=self.kwargs['pk'])
        return super().dispatch(request, *args, **kwargs)

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['imageset'] = self.imageset

        # Получаем результаты из сессии
        batch_results = self.request.session.get('batch_results', {})
        if batch_results.get('imageset_id') == self.imageset.id:
            context.update(batch_results)
            # Очищаем результаты из сессии
            if 'batch_results' in self.request.session:
                del self.request.session['batch_results']
        else:
            # Если нет результатов в сессии, получаем последние результаты из БД
            latest_inferences = InferencedImage.objects.filter(
                orig_image__image_set=self.imageset
            ).filter(detection_info__isnull=False).order_by('-detection_timestamp')[:20]

            detected_images = []
            for inf in latest_inferences:
                if inf.detection_info:
                    # ИСПРАВЛЕНИЕ: используем get_imageurl без скобок
                    detected_images.append({
                        'id': inf.orig_image.id,
                        'name': inf.orig_image.name,
                        'url': inf.orig_image.get_imageurl,
                        'inference_url': inf.inf_image_path,
                        'detections_count': len(inf.detection_info),
                        'classes': list(set([d['class'] for d in inf.detection_info]))
                    })

            context.update({
                'detected_images': detected_images,
                'detected_count': len(detected_images),
            })

        return context


class ImagesDeleteUrl(LoginRequiredMixin, DeleteView):
    model = ImageFile

    def get_success_url(self):
        qs = self.get_object()
        return qs.get_delete_url()
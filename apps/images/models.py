import os
import hashlib
from PIL import Image

from django.db import models
from django.conf import settings
from django.utils.translation import gettext_lazy as _
from django.urls import reverse
from django.core.files.storage import default_storage

from config.models import CreationModificationDateBase


def imageset_upload_images_path(instance, filename):
    """Путь для сохранения изображений в наборе (старая функция для совместимости с миграциями)."""
    return f'imagesets/{instance.image_set.user.username}/{instance.image_set.name}/{filename}'


def imageset_directory_path(instance, filename):
    """Путь для сохранения изображений в наборе."""
    return f'imagesets/{instance.image_set.user.username}/{instance.image_set.name}/{filename}'


def resize_image(image_path, size=(640, 640)):
    """Изменяет размер изображения до указанного размера."""
    try:
        with Image.open(image_path) as img:
            # Конвертируем в RGB если необходимо
            if img.mode in ('RGBA', 'LA', 'P'):
                img = img.convert('RGB')

            # Изменяем размер с сохранением пропорций
            img.thumbnail(size, Image.Resampling.LANCZOS)

            # Создаем новое изображение с белым фоном
            new_img = Image.new('RGB', size, (255, 255, 255))

            # Вставляем изображение по центру
            x = (size[0] - img.size[0]) // 2
            y = (size[1] - img.size[1]) // 2
            new_img.paste(img, (x, y))

            # Сохраняем
            new_img.save(image_path, format='JPEG', quality=90)

    except Exception as e:
        print(f"Ошибка при изменении размера изображения: {e}")


def calculate_image_hash(image_path):
    """Вычисляет хэш изображения для поиска дубликатов."""
    try:
        hasher = hashlib.md5()
        with open(image_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception as e:
        print(f"Ошибка при вычислении хэша: {e}")
        return None


class ImageSet(CreationModificationDateBase):
    """Модель набора изображений."""

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='imagesets',
        verbose_name=_('Пользователь')
    )

    name = models.CharField(
        max_length=100,
        verbose_name=_('Название'),
        help_text=_('Название набора изображений')
    )

    description = models.TextField(
        blank=True,
        verbose_name=_('Описание'),
        help_text=_('Описание набора изображений')
    )

    public = models.BooleanField(
        default=False,
        verbose_name=_('Публичный'),
        help_text=_('Сделать набор доступным для всех пользователей')
    )

    class Meta:
        verbose_name = _('Набор изображений')
        verbose_name_plural = _('Наборы изображений')
        ordering = ['-created']
        constraints = [
            models.UniqueConstraint(
                fields=['user', 'name'],
                name='unique_imageset_per_user'
            )
        ]

    def __str__(self):
        return f"{self.name} ({self.user.username})"

    def get_absolute_url(self):
        return reverse('images:imageset_detail_url', kwargs={'pk': self.pk})

    @property
    def total_images(self):
        """Общее количество изображений в наборе."""
        return self.images.count()

    @property
    def processed_images(self):
        """Количество обработанных изображений."""
        return self.images.filter(is_inferenced=True).count()

    @property
    def unprocessed_images(self):
        """Количество необработанных изображений."""
        return self.images.filter(is_inferenced=False).count()

    @property
    def images_with_objects(self):
        """Количество изображений с найденными объектами."""
        return self.images.filter(detectedimages__isnull=False).distinct().count()


class ImageFile(CreationModificationDateBase):
    """Модель файла изображения."""

    image_set = models.ForeignKey(
        ImageSet,
        on_delete=models.CASCADE,
        related_name='images',
        verbose_name=_('Набор изображений')
    )

    image = models.ImageField(
        upload_to=imageset_directory_path,
        verbose_name=_('Изображение'),
        help_text=_('Файл изображения')
    )

    name = models.CharField(
        max_length=255,
        verbose_name=_('Image Name'),
        help_text=_('Название файла изображения')
    )

    is_inferenced = models.BooleanField(
        default=False,
        verbose_name=_('Обработано'),
        help_text=_('Было ли изображение обработано моделью')
    )

    image_hash = models.CharField(
        max_length=32,
        blank=True,
        null=True,
        verbose_name=_('Хэш изображения'),
        help_text=_('MD5 хэш для поиска дубликатов'),
        db_index=True
    )

    class Meta:
        verbose_name = _('Изображение')
        verbose_name_plural = _('Изображения')
        ordering = ['-created']

    def __str__(self):
        return f"{self.name} ({self.image_set.name})"

    def save(self, *args, **kwargs):
        """Переопределяем save для обработки изображения."""
        # Устанавливаем имя файла если не задано
        if not self.name and self.image:
            self.name = os.path.basename(self.image.name)

        # Сохраняем объект
        super().save(*args, **kwargs)

        # Обрабатываем изображение после сохранения
        if self.image:
            image_path = self.image.path

            # Изменяем размер изображения
            resize_image(image_path, size=(640, 640))

            # Вычисляем хэш изображения
            if not self.image_hash:
                self.image_hash = calculate_image_hash(image_path)
                # Сохраняем без вызова save() чтобы избежать рекурсии
                ImageFile.objects.filter(pk=self.pk).update(image_hash=self.image_hash)

    def delete(self, *args, **kwargs):
        """Переопределяем delete для удаления файла."""
        # Удаляем файл изображения
        if self.image:
            if default_storage.exists(self.image.name):
                default_storage.delete(self.image.name)

        super().delete(*args, **kwargs)

    def get_absolute_url(self):
        return reverse('detectobj:detection_image_detail_url', kwargs={'pk': self.pk})

    @property
    def get_imageurl(self):
        """Возвращает URL изображения."""
        if self.image:
            return self.image.url
        return None

    @property
    def get_imagepath(self):
        """Возвращает путь к файлу изображения."""
        if self.image:
            return self.image.path
        return None

    def get_imgshape(self):
        """Возвращает размеры изображения."""
        try:
            if self.image:
                with Image.open(self.image.path) as img:
                    return f"{img.width}x{img.height}"
        except Exception:
            pass
        return "Unknown"

    @property
    def has_detections(self):
        """Проверяет, есть ли у изображения детекции."""
        return self.detectedimages.exists()

    @property
    def has_manual_annotations(self):
        """Проверяет, есть ли у изображения ручные аннотации."""
        return self.manual_annotations.exists()

    def find_duplicates(self):
        """Находит дубликаты изображения по хэшу."""
        if self.image_hash:
            return ImageFile.objects.filter(
                image_hash=self.image_hash
            ).exclude(pk=self.pk)
        return ImageFile.objects.none()
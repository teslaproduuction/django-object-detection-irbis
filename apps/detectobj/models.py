import os

from django.conf import settings
from django.db import models
from django.utils.translation import gettext_lazy as _
from django.urls import reverse
from config.models import CreationModificationDateBase


class InferencedImage(CreationModificationDateBase):
    orig_image = models.ForeignKey(
        "images.ImageFile",
        on_delete=models.CASCADE,
        related_name="detectedimages",
        help_text="Main Image",
        null=True,
        blank=True
    )

    inf_image_path = models.CharField(max_length=250,
                                      null=True,
                                      blank=True
                                      )

    custom_model = models.ForeignKey("modelmanager.MLModel",
                                     verbose_name="Custom ML Models",
                                     on_delete=models.DO_NOTHING,
                                     null=True,
                                     blank=True,
                                     related_name="detectedimages",
                                     help_text="Machine Learning model for detection",
                                     )
    detection_info = models.JSONField(null=True, blank=True)

    YOLOMODEL_CHOICES = [
        ('yolov8n.pt', 'yolov8n.pt'),  # Nano model
        ('yolov8s.pt', 'yolov8s.pt'),  # Small model
        ('yolov8m.pt', 'yolov8m.pt'),  # Medium model
        ('yolov8l.pt', 'yolov8l.pt'),  # Large model
        ('yolov8x.pt', 'yolov8x.pt'),  # Extra-large model
    ]

    yolo_model = models.CharField(_('YOLOV8 Models'),
                                  max_length=250,
                                  null=True,
                                  blank=True,
                                  choices=YOLOMODEL_CHOICES,
                                  default=YOLOMODEL_CHOICES[0],
                                  help_text="Selected yolo model will download. \
                                 Requires an active internet connection."
                                  )

    model_conf = models.DecimalField(_('Model confidence'),
                                     decimal_places=2,
                                     max_digits=4,
                                     null=True,
                                     blank=True)

    # Add a new field to store detection timestamp for multiple detection records of the same image
    detection_timestamp = models.DateTimeField(auto_now_add=True,
                                               verbose_name=_('Detection Timestamp'))

    class Meta:
        # Order by most recent detection first
        ordering = ['-detection_timestamp']


class ManualAnnotation(CreationModificationDateBase):
    """Модель для хранения ручных аннотаций изображений."""

    image = models.ForeignKey(
        "images.ImageFile",
        on_delete=models.CASCADE,
        related_name="manual_annotations",
        help_text=_("Изображение для ручной разметки"),
        verbose_name=_("Изображение")
    )

    # Координаты рамки (нормализованные от 0 до 1)
    x_center = models.FloatField(_("X центра"), help_text=_("Нормализованная координата X центра бокса (от 0 до 1)"))
    y_center = models.FloatField(_("Y центра"), help_text=_("Нормализованная координата Y центра бокса (от 0 до 1)"))
    width = models.FloatField(_("Ширина"), help_text=_("Нормализованная ширина бокса (от 0 до 1)"))
    height = models.FloatField(_("Высота"), help_text=_("Нормализованная высота бокса (от 0 до 1)"))

    # Класс объекта
    class_name = models.CharField(_("Класс объекта"), max_length=100, help_text=_("Название класса объекта"))
    confidence = models.FloatField(_("Достоверность"), default=1.0,
                                   help_text=_("Степень уверенности (для ручной разметки обычно 1)"))

    # Признак, указывающий что аннотация создана вручную
    is_manual = models.BooleanField(_("Ручная разметка"), default=True)

    # Пользователь, создавший аннотацию
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        null=True,
        related_name="annotations",
        verbose_name=_("Создано пользователем")
    )

    class Meta:
        verbose_name = _("Ручная аннотация")
        verbose_name_plural = _("Ручные аннотации")
        ordering = ['-created']

    def __str__(self):
        return f"{self.class_name} на {self.image.name}"

    def get_absolute_url(self):
        return reverse("detectobj:detection_image_detail_url", kwargs={"pk": self.image.id})

    def to_detection_format(self):
        """Преобразование аннотации в формат, совместимый с InferencedImage.detection_info"""
        return {
            "class": self.class_name,
            "confidence": self.confidence,
            "x": self.x_center,
            "y": self.y_center,
            "width": self.width,
            "height": self.height,
            "is_manual": True  # Добавляем маркер для ручных аннотаций
        }
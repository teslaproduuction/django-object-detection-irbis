import os

from django.conf import settings
from django.db import models
from django.utils.translation import gettext_lazy as _
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
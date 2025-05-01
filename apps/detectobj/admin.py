from django.contrib import admin
from .models import InferencedImage, ManualAnnotation


@admin.register(InferencedImage)
class InferencedImageAdmin(admin.ModelAdmin):
    list_display = ["orig_image", "inf_image_path",
                    "model_conf", "custom_model"]


@admin.register(ManualAnnotation)
class ManualAnnotationAdmin(admin.ModelAdmin):
    list_display = ["image", "class_name", "created_by", "created", "is_manual"]
    list_filter = ["class_name", "created_by", "is_manual"]
    search_fields = ["class_name", "created_by__username", "image__name"]
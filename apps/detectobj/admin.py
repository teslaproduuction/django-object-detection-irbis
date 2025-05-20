from django.contrib import admin
from unfold.admin import ModelAdmin
from .models import InferencedImage, ManualAnnotation


@admin.register(InferencedImage)
class InferencedImageAdmin(ModelAdmin):
    list_display = ["orig_image", "inf_image_path", "model_conf", "custom_model", "detection_timestamp"]
    list_filter = ["custom_model", "yolo_model"]
    search_fields = ["orig_image__name"]
    date_hierarchy = "detection_timestamp"

    # Unfold specific features
    actions_list = ["mark_as_reviewed"]

    @admin.action(description="Mark selected images as reviewed")
    def mark_as_reviewed(self, request, queryset):
        # Placeholder action for demonstration
        queryset.update(detection_info={"reviewed": True})
        self.message_user(request, f"{queryset.count()} images marked as reviewed.")


@admin.register(ManualAnnotation)
class ManualAnnotationAdmin(ModelAdmin):
    list_display = ["image", "class_name", "created_by", "created", "is_manual", "confidence"]
    list_filter = ["class_name", "created_by", "is_manual"]
    search_fields = ["class_name", "created_by__username", "image__name"]

    # Unfold specific features
    list_editable = ["confidence"]

    fieldsets = [
        (None, {"fields": ["image", "class_name", "created_by"]}),
        ("Annotation Details", {"fields": ["x_center", "y_center", "width", "height", "confidence"]}),
        ("Status", {"fields": ["is_manual"]}),
    ]
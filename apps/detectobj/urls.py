from django.urls import path
from . import views

app_name = "detectobj"

urlpatterns = [
    path("<int:pk>/selected_image/",
         views.InferenceImageDetectionView.as_view(),
         name="detection_image_detail_url"
         ),
    # Добавляем URL для ручной разметки
    path("<int:pk>/manual_annotation/",
         views.ManualAnnotationView.as_view(),
         name="manual_annotation_url"
         ),
    # URL для сохранения аннотаций через AJAX
    path("save_annotation/",
         views.SaveAnnotationView.as_view(),
         name="save_annotation_url"
         ),
]
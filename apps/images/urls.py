from django.urls import path
from . import views

app_name = "images"

urlpatterns = [
    # ImageSet URLs
    path('imageset/list/',
         views.ImageSetListView.as_view(),
         name='imageset_list_url'),

    path('imageset/create/',
         views.ImageSetCreateView.as_view(),
         name='imageset_create_url'),

    path('imageset/<int:pk>/',
         views.ImageSetDetailView.as_view(),
         name='imageset_detail_url'),

    path('imageset/<int:pk>/update/',
         views.ImageSetUpdateView.as_view(),
         name='imageset_update_url'),

    # URL для удаления набора изображений
    path('imageset/<int:pk>/delete/',
         views.ImageSetDeleteView.as_view(),
         name='imageset_delete_url'),

    # URL для пакетного распознавания
    path('imageset/<int:pk>/batch_detection/',
         views.BatchDetectionView.as_view(),
         name='batch_detection_url'),

    # URL для результатов пакетного распознавания
    path('imageset/<int:pk>/batch_detection/results/',
         views.BatchDetectionResultsView.as_view(),
         name='batch_detection_results_url'),

    # ImageFile URLs
    path('imageset/<int:imageset_id>/upload/',
         views.ImageFileCreateView.as_view(),
         name='upload_images_url'),

    path('imageset/<int:imageset_id>/images/',
         views.ImageFileListView.as_view(),
         name='imagefile_list_url'),

    path('image/<int:pk>/delete/',
         views.ImageFileDeleteView.as_view(),
         name='imagefile_delete_url'),

    # Detection URL (редирект на detectobj приложение)
    path('imageset/<int:imageset_id>/image/<int:image_id>/detect/',
         views.ImageFileListView.as_view(),
         name='detect_object_url'),
]
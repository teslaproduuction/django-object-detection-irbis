"""URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
"""
from django.contrib import admin
from django.conf import settings
from django.conf.urls.static import static
from django.urls import path, include
from django.conf.urls.i18n import i18n_patterns

from . import views


# URL для функций, не требующих локализации
urlpatterns = [
    path('i18n/', include('django.conf.urls.i18n')),  # URL для смены языка
]

# Добавляем основные URL, которые будут переведены
urlpatterns += i18n_patterns(
    path('admin/', admin.site.urls),
    path('', views.HomeTemplateView.as_view(), name="home_url"),
    path('users/', include('users.urls', namespace='users')),
    path('detectobj/', include("detectobj.urls", namespace="detectobj")),
    path('modelmanager/', include("modelmanager.urls", namespace="modelmanager")),
    path('images/', include("images.urls", namespace="images")),
    prefix_default_language=True  # Не добавлять префикс для языка по умолчанию
)

# Статические и медиа файлы в режиме разработки
if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL,
                          document_root=settings.STATIC_ROOT)
    urlpatterns += static(settings.MEDIA_URL,
                          document_root=settings.MEDIA_ROOT)


if settings.DEBUG and 'debug_toolbar' in settings.INSTALLED_APPS:
    import debug_toolbar
    urlpatterns += [
        path('__debug__/', include(debug_toolbar.urls)),
    ]
from django.contrib import admin
from django.utils.html import format_html
from django.urls import reverse
from django.utils.translation import gettext_lazy as _
from unfold.admin import ModelAdmin

from .models import ImageSet, ImageFile


@admin.register(ImageSet)
class ImageSetAdmin(ModelAdmin):
    """Админ для наборов изображений."""

    list_display = [
        'name', 'user', 'total_images_display', 'processed_images_display',
        'public', 'created', 'modified'
    ]
    list_filter = ['public', 'user', 'created']
    search_fields = ['name', 'description', 'user__username']
    readonly_fields = ['created', 'modified']
    date_hierarchy = 'created'

    fieldsets = [
        (None, {
            'fields': ['name', 'description', 'user', 'public']
        }),
        (_('Статистика'), {
            'fields': ['total_images_display', 'processed_images_display', 'images_with_objects_display'],
            'classes': ['collapse']
        }),
        (_('Даты'), {
            'fields': ['created', 'modified'],
            'classes': ['collapse']
        }),
    ]

    # Unfold specific features
    actions_list = ['make_public', 'make_private', 'reset_processing_status']

    def total_images_display(self, obj):
        """Отображение общего количества изображений."""
        count = obj.total_images
        if count > 0:
            url = reverse('admin:images_imagefile_changelist') + f'?image_set__id__exact={obj.id}'
            return format_html(
                '<a href="{}">{} изображений</a>',
                url, count
            )
        return '0 изображений'

    total_images_display.short_description = _('Всего изображений')

    def processed_images_display(self, obj):
        """Отображение количества обработанных изображений."""
        processed = obj.processed_images
        total = obj.total_images
        if total > 0:
            percentage = (processed / total) * 100
            color = 'green' if percentage > 80 else 'orange' if percentage > 50 else 'red'
            return format_html(
                '<span style="color: {};">{}/{} ({:.1f}%)</span>',
                color, processed, total, percentage
            )
        return '0/0 (0%)'

    processed_images_display.short_description = _('Обработано')

    def images_with_objects_display(self, obj):
        """Отображение количества изображений с объектами."""
        with_objects = obj.images_with_objects
        return format_html(
            '<span class="badge bg-info">{}</span>',
            with_objects
        )

    images_with_objects_display.short_description = _('С объектами')

    @admin.action(description=_('Сделать публичными'))
    def make_public(self, request, queryset):
        """Делает наборы публичными."""
        updated = queryset.update(public=True)
        self.message_user(
            request,
            _('%(count)d наборов сделано публичными.') % {'count': updated}
        )

    @admin.action(description=_('Сделать приватными'))
    def make_private(self, request, queryset):
        """Делает наборы приватными."""
        updated = queryset.update(public=False)
        self.message_user(
            request,
            _('%(count)d наборов сделано приватными.') % {'count': updated}
        )

    @admin.action(description=_('Сбросить статус обработки'))
    def reset_processing_status(self, request, queryset):
        """Сбрасывает статус обработки для всех изображений в выбранных наборах."""
        total_updated = 0
        for imageset in queryset:
            updated = imageset.images.update(is_inferenced=False)
            total_updated += updated

        self.message_user(
            request,
            _('Статус обработки сброшен для %(count)d изображений.') % {'count': total_updated}
        )


class ImageFileInline(admin.TabularInline):
    """Inline для изображений в наборе."""
    model = ImageFile
    extra = 0
    readonly_fields = ['name', 'image_preview', 'is_inferenced', 'image_hash', 'created']
    fields = ['image_preview', 'name', 'is_inferenced', 'image_hash', 'created']

    def image_preview(self, obj):
        """Превью изображения."""
        if obj.image:
            return format_html(
                '<img src="{}" style="width: 100px; height: 100px; object-fit: cover;" />',
                obj.get_imageurl
            )
        return '-'

    image_preview.short_description = _('Превью')


@admin.register(ImageFile)
class ImageFileAdmin(ModelAdmin):
    """Админ для изображений."""

    list_display = [
        'image_preview_small', 'name', 'image_set', 'is_inferenced',
        'has_detections_display', 'has_annotations_display', 'created'
    ]
    list_filter = [
        'is_inferenced', 'image_set', 'image_set__user',
        'created', 'image_set__public'
    ]
    search_fields = ['name', 'image_set__name', 'image_set__user__username']
    readonly_fields = [
        'name', 'image_preview', 'image_hash', 'get_imgshape_display',
        'created', 'modified', 'duplicates_display'
    ]
    date_hierarchy = 'created'

    fieldsets = [
        (None, {
            'fields': ['image_set', 'image', 'name']
        }),
        (_('Статус'), {
            'fields': ['is_inferenced', 'has_detections_display', 'has_annotations_display']
        }),
        (_('Информация об изображении'), {
            'fields': ['image_preview', 'get_imgshape_display', 'image_hash', 'duplicates_display'],
            'classes': ['collapse']
        }),
        (_('Даты'), {
            'fields': ['created', 'modified'],
            'classes': ['collapse']
        }),
    ]

    # Unfold specific features
    actions_list = ['mark_as_processed', 'mark_as_unprocessed', 'find_duplicates']

    def image_preview_small(self, obj):
        """Маленькое превью для списка."""
        if obj.image:
            return format_html(
                '<img src="{}" style="width: 50px; height: 50px; object-fit: cover;" />',
                obj.get_imageurl
            )
        return '-'

    image_preview_small.short_description = _('Превью')

    def image_preview(self, obj):
        """Большое превью для формы."""
        if obj.image:
            return format_html(
                '<img src="{}" style="max-width: 300px; max-height: 300px;" />',
                obj.get_imageurl
            )
        return '-'

    image_preview.short_description = _('Изображение')

    def has_detections_display(self, obj):
        """Отображение наличия детекций."""
        if obj.has_detections:
            return format_html('<span style="color: green;">✓</span>')
        return format_html('<span style="color: red;">✗</span>')

    has_detections_display.short_description = _('Детекции')
    has_detections_display.boolean = True

    def has_annotations_display(self, obj):
        """Отображение наличия ручных аннотаций."""
        if obj.has_manual_annotations:
            return format_html('<span style="color: green;">✓</span>')
        return format_html('<span style="color: red;">✗</span>')

    has_annotations_display.short_description = _('Аннотации')
    has_annotations_display.boolean = True

    def get_imgshape_display(self, obj):
        """Отображение размеров изображения."""
        return obj.get_imgshape()

    get_imgshape_display.short_description = _('Размеры')

    def duplicates_display(self, obj):
        """Отображение дубликатов."""
        duplicates = obj.find_duplicates()
        count = duplicates.count()
        if count > 0:
            return format_html(
                '<span style="color: orange;">{} дубликатов</span>',
                count
            )
        return _('Дубликатов нет')

    duplicates_display.short_description = _('Дубликаты')

    @admin.action(description=_('Отметить как обработанные'))
    def mark_as_processed(self, request, queryset):
        """Отмечает изображения как обработанные."""
        updated = queryset.update(is_inferenced=True)
        self.message_user(
            request,
            _('%(count)d изображений отмечено как обработанные.') % {'count': updated}
        )

    @admin.action(description=_('Отметить как необработанные'))
    def mark_as_unprocessed(self, request, queryset):
        """Отмечает изображения как необработанные."""
        updated = queryset.update(is_inferenced=False)
        self.message_user(
            request,
            _('%(count)d изображений отмечено как необработанные.') % {'count': updated}
        )

    @admin.action(description=_('Найти дубликаты'))
    def find_duplicates(self, request, queryset):
        """Находит дубликаты среди выбранных изображений."""
        duplicates_count = 0
        for image in queryset:
            duplicates = image.find_duplicates()
            duplicates_count += duplicates.count()

        self.message_user(
            request,
            _('Найдено %(count)d дубликатов.') % {'count': duplicates_count}
        )


# Настраиваем заголовки админки
admin.site.site_header = _('Система обнаружения объектов')
admin.site.site_title = _('Админ-панель')
admin.site.index_title = _('Управление системой')
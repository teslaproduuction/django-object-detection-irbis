from django import forms
from django.db import models
from django.utils.translation import gettext_lazy as _

from .models import ImageSet, ImageFile
from detectobj.models import InferencedImage
from modelmanager.models import MLModel


class ImageSetForm(forms.ModelForm):
    """Форма для создания и редактирования наборов изображений."""

    class Meta:
        model = ImageSet
        fields = ['name', 'description', 'public']
        widgets = {
            'name': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': _('Введите название набора')
            }),
            'description': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 3,
                'placeholder': _('Описание набора изображений')
            }),
            'public': forms.CheckboxInput(attrs={
                'class': 'form-check-input'
            })
        }


class ImageFileForm(forms.ModelForm):
    """Форма для загрузки изображений."""

    class Meta:
        model = ImageFile
        fields = ['image']
        widgets = {
            'image': forms.FileInput(attrs={
                'class': 'form-control',
                'accept': 'image/*',
                'multiple': True
            })
        }


class BatchDetectionForm(forms.Form):
    """Форма для выбора модели для пакетного распознавания."""

    MODEL_TYPE_CHOICES = [
        ('yolo', _('Модель YOLO')),
        ('custom', _('Кастомная модель')),
    ]

    model_type = forms.ChoiceField(
        choices=MODEL_TYPE_CHOICES,
        widget=forms.RadioSelect(attrs={
            'class': 'form-check-input'
        }),
        label=_('Тип модели'),
        initial='yolo'
    )

    yolo_model = forms.ChoiceField(
        choices=InferencedImage.YOLOMODEL_CHOICES,
        widget=forms.Select(attrs={
            'class': 'form-select'
        }),
        label=_('YOLO модель'),
        required=False,
        initial=InferencedImage.YOLOMODEL_CHOICES[0][0]
    )

    custom_model = forms.ModelChoiceField(
        queryset=MLModel.objects.none(),
        widget=forms.Select(attrs={
            'class': 'form-select'
        }),
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
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'step': '0.01',
            'min': '0.25',
            'max': '1.0'
        }),
        help_text=_("Порог уверенности модели для предсказаний (от 0.25 до 1.0)."),
    )

    def __init__(self, *args, **kwargs):
        user = kwargs.pop('user', None)
        super().__init__(*args, **kwargs)

        if user:
            # Показываем модели пользователя и публичные модели
            self.fields['custom_model'].queryset = MLModel.objects.filter(
                models.Q(uploader=user) | models.Q(public=True)
            ).order_by('name')

    def clean(self):
        """Проверяем, что выбрана хотя бы одна модель."""
        cleaned_data = super().clean()
        model_type = cleaned_data.get('model_type')
        yolo_model = cleaned_data.get('yolo_model')
        custom_model = cleaned_data.get('custom_model')

        if model_type == 'yolo' and not yolo_model:
            raise forms.ValidationError(_('Выберите YOLO модель для распознавания'))

        if model_type == 'custom' and not custom_model:
            raise forms.ValidationError(_('Выберите кастомную модель для распознавания'))

        return cleaned_data


class ImageSearchForm(forms.Form):
    """Форма для поиска и фильтрации изображений."""

    FILTER_CHOICES = [
        ('all', _('Все изображения')),
        ('processed', _('Обработанные')),
        ('unprocessed', _('Необработанные')),
        ('with_objects', _('С найденными объектами')),
        ('without_objects', _('Без объектов')),
    ]

    search = forms.CharField(
        required=False,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': _('Поиск по имени файла...'),
        }),
        label=_('Поиск')
    )

    filter_type = forms.ChoiceField(
        choices=FILTER_CHOICES,
        required=False,
        initial='all',
        widget=forms.Select(attrs={
            'class': 'form-select'
        }),
        label=_('Фильтр')
    )

    order_by = forms.ChoiceField(
        choices=[
            ('-created', _('Сначала новые')),
            ('created', _('Сначала старые')),
            ('name', _('По имени (А-Я)')),
            ('-name', _('По имени (Я-А)')),
        ],
        required=False,
        initial='-created',
        widget=forms.Select(attrs={
            'class': 'form-select'
        }),
        label=_('Сортировка')
    )
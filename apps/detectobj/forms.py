from django import forms
from django.utils.translation import gettext_lazy as _
from .models import InferencedImage, ManualAnnotation


class InferencedImageForm(forms.ModelForm):
    model_conf = forms.DecimalField(label=_("Порог уверенности модели"),
                                    max_value=1,
                                    min_value=0.25,
                                    max_digits=3,
                                    decimal_places=2,
                                    initial=0.45,
                                    help_text=_("Порог уверенности модели для предсказаний."),
                                    )

    class Meta:
        model = InferencedImage
        fields = ('custom_model', 'model_conf')


class YoloModelForm(forms.ModelForm):
    model_conf = forms.DecimalField(label=_("Порог уверенности модели"),
                                    max_value=1,
                                    min_value=0.25,
                                    max_digits=3,
                                    decimal_places=2,
                                    initial=0.45,
                                    help_text=_("Порог уверенности модели для предсказаний."),
                                    )

    class Meta:
        model = InferencedImage
        fields = ('yolo_model', 'model_conf')


class ManualAnnotationForm(forms.ModelForm):
    """Форма для ручной разметки изображений."""

    class Meta:
        model = ManualAnnotation
        fields = ['class_name', 'x_center', 'y_center', 'width', 'height']
        widgets = {
            'x_center': forms.HiddenInput(),
            'y_center': forms.HiddenInput(),
            'width': forms.HiddenInput(),
            'height': forms.HiddenInput(),
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Добавляем поле для выбора класса из списка
        self.fields['class_name'] = forms.CharField(
            label=_("Класс объекта"),
            widget=forms.TextInput(attrs={'class': 'form-control', 'list': 'class-list'}),
            help_text=_("Введите или выберите класс объекта")
        )


class ManualAnnotationFormSet(forms.BaseFormSet):
    """Набор форм для одновременного добавления нескольких аннотаций."""

    def __init__(self, *args, **kwargs):
        self.image = kwargs.pop('image', None)
        self.user = kwargs.pop('user', None)
        super().__init__(*args, **kwargs)

    def save(self):
        """Сохранение всех форм в наборе."""
        annotations = []
        for form in self.forms:
            if form.is_valid() and form.cleaned_data and not form.cleaned_data.get('DELETE', False):
                annotation = form.save(commit=False)
                annotation.image = self.image
                annotation.created_by = self.user
                annotation.is_manual = True
                annotation.save()
                annotations.append(annotation)
        return annotations


# Фабрика для создания набора форм
ManualAnnotationFormSetFactory = forms.formset_factory(
    ManualAnnotationForm,
    formset=ManualAnnotationFormSet,
    extra=1,
    can_delete=True
)
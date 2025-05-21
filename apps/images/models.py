import os
import hashlib
from PIL import Image as I

from django.db import models
from django.conf import settings
from django.urls import reverse
from django.utils.translation import gettext_lazy as _
from config.models import CreationModificationDateBase


class ImageSet(CreationModificationDateBase):
    name = models.CharField(max_length=100,
                            help_text="eg. Delhi-trip, Tajmahal, flowers"
                            )
    description = models.TextField()
    user = models.ForeignKey(settings.AUTH_USER_MODEL,
                             related_name='imagesets',
                             on_delete=models.CASCADE
                             )
    dirpath = models.CharField(max_length=150, null=True, blank=True)
    public = models.BooleanField(default=False)

    class Meta:
        constraints = [models.UniqueConstraint(
            fields=['user', 'name'],
            name='unique_imageset_by_user')]

    def __str__(self):
        return f'{self.name.capitalize()}'

    def get_dirpath(self):
        return os.path.join(self.user.username, self.name)

    def get_absolute_url(self):
        return reverse("images:imageset_detail_url", kwargs={"pk": self.pk})


def imageset_upload_images_path(instance, filename):
    return f'{instance.image_set.dirpath}/images/{filename}'


class ImageFile(models.Model):
    name = models.CharField(_('Image Name'), max_length=150, null=True)
    image_set = models.ForeignKey('images.ImageSet',
                                  related_name="images",
                                  on_delete=models.CASCADE,
                                  help_text="Image Set of the uploading images"
                                  )
    image = models.ImageField(upload_to=imageset_upload_images_path)

    is_inferenced = models.BooleanField(default=False)

    # Add a field to store image hash
    image_hash = models.CharField(max_length=64, blank=True, null=True, db_index=True)

    def __str__(self):
        return self.name

    @property
    def get_imageurl(self):
        return self.image.url

    @property
    def get_imagepath(self):
        return self.image.path

    @property
    def get_filename(self):
        return os.path.split(self.image.url)[-1]

    @property
    def get_imgshape(self):
        im = I.open(self.get_imagepath)
        return im.size

    def get_delete_url(self):
        return reverse("images:images_list_url", kwargs={"pk": self.image_set.id})

    def save(self, *args, **kwargs):
        # Generate hash for the image if not already set
        if not self.image_hash and self.image:
            try:
                with open(self.image.path, 'rb') as f:
                    file_hash = hashlib.sha256()
                    for chunk in iter(lambda: f.read(4096), b''):
                        file_hash.update(chunk)
                    self.image_hash = file_hash.hexdigest()
            except Exception as e:
                # If image file doesn't exist yet, we'll generate the hash after it's saved
                print(f"Unable to generate image hash: {e}")

        # Call the parent class save method
        super().save(*args, **kwargs)

        # If hash wasn't generated before saving and the image file now exists, generate it
        if not self.image_hash and self.image:
            try:
                with open(self.get_imagepath, 'rb') as f:
                    file_hash = hashlib.sha256()
                    for chunk in iter(lambda: f.read(4096), b''):
                        file_hash.update(chunk)
                    self.image_hash = file_hash.hexdigest()
                # Save again to store the hash, but avoid recursion
                ImageFile.objects.filter(pk=self.pk).update(image_hash=self.image_hash)
            except Exception as e:
                print(f"Unable to generate image hash after save: {e}")

        # Convert image to 640px before saving.
        img = I.open(self.get_imagepath)
        if img.height > 640 or img.width > 640:
            output_size = (640, 640)
            img.thumbnail(output_size)
            img.save(self.get_imagepath)
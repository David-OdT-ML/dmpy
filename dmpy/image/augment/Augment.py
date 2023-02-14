from albumentations import (
    HorizontalFlip, VerticalFlip, 
    ShiftScaleRotate, OpticalDistortion, GridDistortion, ElasticTransform, 
    RandomBrightnessContrast, 
    Sharpen, IAAEmboss, Flip, OneOf, Compose
)

class Augment(object):


    def __init__(self) -> None:
        pass


    def composeAll(self,probabilities=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
                        shift_limit=0.0625,
                        scale_limit=0.2,
                        rotate_limit=45.0,
                        ealpha=400, 
                        esigma=400 * 0.05, 
                        ealpha_affine=400 * 0.03,
                        gdistort_limit=0.3, 
                        gnumsteps=5,
                        odistort_limit=0.05, 
                        oshift_limit=0.05,
                        salpha=(0.2, 0.5), 
                        slightness=(0.5, 1)
                        ):
        transform = Compose([
            HorizontalFlip(p=probabilities[0]),
            VerticalFlip(p=probabilities[1]),
            ShiftScaleRotate(p=probabilities[2], shift_limit=shift_limit, scale_limit=scale_limit, rotate_limit=rotate_limit),
            GridDistortion(p=probabilities[3], distort_limit=gdistort_limit, num_steps=gnumsteps),
            OpticalDistortion(p=probabilities[4], distort_limit=odistort_limit, shift_limit=oshift_limit),
            ElasticTransform(p=probabilities[5], alpha=ealpha, sigma=esigma, alpha_affine=ealpha_affine),
            Sharpen(p=probabilities[6], alpha=salpha, lightness=slightness)
            ])
        return transform


    def getTransformed(self, transform, imagePatch, targetPatch):
        transformed = transform(image=imagePatch, masks=targetPatch)
        return transformed['image'], transformed['mask']


    def hFlip(self, imagePatch, targetPatch, probability=1.0):
        a = HorizontalFlip(p=probability)
        aug = a(image=imagePatch, mask=targetPatch)
        return aug['image'], aug['mask']


    def VFlip(self, imagePatch, targetPatch, probability=1.0):
        a = VerticalFlip(p=probability)
        aug = a(image=imagePatch, mask=targetPatch)
        return aug['image'], aug['mask']


    def Shift(self, imagePatch, targetPatch, probability=1.0, shift_limit=0.0625):
        a = ShiftScaleRotate(p=probability, shift_limit=shift_limit)
        aug = a(image=imagePatch, mask=targetPatch)
        return aug['image'], aug['mask']


    def Scale(self, imagePatch, targetPatch, probability=1.0, scale_limit=0.2):
        a = ShiftScaleRotate(p=probability, scale_limit=scale_limit)
        aug = a(image=imagePatch, mask=targetPatch)
        return aug['image'], aug['mask']


    def Rotate(self, imagePatch, targetPatch, probability=1.0, rotate_limit=45.0):
        a = ShiftScaleRotate(p=probability, rotate_limit=rotate_limit)
        aug = a(image=imagePatch, mask=targetPatch)
        return aug['image'], aug['mask']


    def ShiftScaleRotate(self, imagePatch, targetPatch, probability=1.0, shift_limit=0.0625, scale_limit=0.2, rotate_limit=45.0):
        a = ShiftScaleRotate(p=probability, shift_limit=shift_limit, scale_limit=scale_limit, rotate_limit=rotate_limit)
        aug = a(image=imagePatch, mask=targetPatch)
        return aug['image'], aug['mask']


    def GridDistort(self, imagePatch, targetPatch, probability=1.0, gdistort_limit=0.3, gnumsteps=15):
        a = GridDistortion(p=probability, distort_limit=gdistort_limit, num_steps=gnumsteps)
        aug = a(image=imagePatch, mask=targetPatch)
        return aug['image'], aug['mask']


    def OpticalDistort(self, imagePatch, targetPatch, probability=1.0, odistort_limit=0.05, oshift_limit=0.05):
        a = OpticalDistortion(p=probability, distort_limit=odistort_limit, shift_limit=oshift_limit)
        aug = a(image=imagePatch, mask=targetPatch)
        return aug['image'], aug['mask']


    def ElasticTransform(self, imagePatch, targetPatch, probability=1.0, ealpha=400, esigma=400 * 0.05, ealpha_affine=400 * 0.03):
        a = ElasticTransform(p=probability, alpha=ealpha, sigma=esigma, alpha_affine=ealpha_affine)
        aug = a(image=imagePatch, mask=targetPatch)
        return aug['image'], aug['mask']


    def Sharpen(self, imagePatch, targetPatch, probability=1.0, salpha=(0.2, 0.5), slightness=(0.5, 1)):
        a = Sharpen(p=probability, alpha=salpha, lightness=slightness)
        aug = a(image=imagePatch, mask=targetPatch)
        return aug['image'], aug['mask']



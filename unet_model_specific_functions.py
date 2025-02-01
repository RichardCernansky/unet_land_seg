
def dice_coef(y_true, y_pred, smooth=1):
    """
    Dice coefficient loss function.
    """
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])  # Adjust axis for your data format
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    dice = K.mean((2. * intersection + smooth) / (union + smooth), axis=0)
    return dice

def dice_loss(y_true, y_pred):
    """
    Dice loss (1 - dice_coef).  Minimizing this is equivalent to maximizing Dice.
    """
    return 1 - dice_coef(y_true, y_pred)



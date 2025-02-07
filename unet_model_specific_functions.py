
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

# Weighted Focal Loss Function
def weighted_focal_loss(alpha=[0.2, 0.8], gamma=2.0):
    """
    Weighted Focal Loss for binary segmentation.

    Parameters:
        alpha (list): List of class weights [weight for class 0, weight for class 1].
        gamma (float): Focusing parameter to down-weight easy examples.
    """
    def loss(y_true, y_pred):
        # Clip predictions to prevent log(0) errors
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)

        # Compute cross-entropy loss
        cross_entropy = -y_true * K.log(y_pred) - (1 - y_true) * K.log(1 - y_pred)

        # Apply focal weight
        weight = alpha[1] * y_true * K.pow(1 - y_pred, gamma) + alpha[0] * (1 - y_true) * K.pow(y_pred, gamma)
        focal_loss = weight * cross_entropy

        return K.mean(focal_loss)

    return loss

# Focal Loss Function
def focal_loss(alpha=0.25, gamma=2.0):
    """
    Focal Loss for binary segmentation.
    Parameters:
        alpha (float): Balancing factor for class imbalance.
        gamma (float): Focusing parameter to down-weight easy examples.
    """
    def loss(y_true, y_pred):
        # Clip predictions to prevent log(0) errors
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)

        # Compute cross-entropy loss
        cross_entropy = -y_true * K.log(y_pred) - (1 - y_true) * K.log(1 - y_pred)

        # Apply focal weight
        weight = alpha * y_true * K.pow(1 - y_pred, gamma) + (1 - alpha) * (1 - y_true) * K.pow(y_pred, gamma)
        focal_loss = weight * cross_entropy

        return K.mean(focal_loss)

    return loss



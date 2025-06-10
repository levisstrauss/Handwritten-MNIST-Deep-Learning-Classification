import torch.optim as optim


def get_optimizer_and_scheduler(model, optimizer_type='adamw', lr=0.001,
                                scheduler_type='onecycle', epochs=15,
                                steps_per_epoch=None):
    """
    Get optimizer and scheduler configuration.

    Args:
        model: PyTorch model
        optimizer_type (str): 'adam', 'adamw', or 'sgd'
        lr (float): Learning rate
        scheduler_type (str): 'step', 'onecycle', or None
        epochs (int): Number of training epochs
        steps_per_epoch (int): Steps per epoch for OneCycleLR

    Returns:
        tuple: (optimizer, scheduler)
    """

    # Select optimizer
    if optimizer_type.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    elif optimizer_type.lower() == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    elif optimizer_type.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_type}")

    # Select scheduler
    scheduler = None
    if scheduler_type == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)
    elif scheduler_type == 'onecycle' and steps_per_epoch:
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr * 10, epochs=epochs, steps_per_epoch=steps_per_epoch
        )

    return optimizer, scheduler

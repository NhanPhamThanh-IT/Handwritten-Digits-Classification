class SystemVariables:
    TITLE = "Welcome on your first sketch recognition app!"
    HEAD = (
    "<center>"
    "The robot was trained to classify numbers (from 0 to 9). To test it, write your number in the space provided."
    "</center>"
    )
    MODELS = 'models/model.keras'
    IMG_SIZE = 28
    LABELS = [str(i) for i in range(10)]
import torch
import torchvision
import argparse
import model_builder


# Creating a parser
parser = argparse.ArgumentParser()

# Get an image path
parser.add_argument("--image_path", help="target image path to predict on")

# Get a model path
parser.add_argument("--model_path",
                    default="models/tiny_vgg_model_v1.pth",
                    type=str, 
                    help="target model to use for prediction filepath")

args = parser.parse_args()

# Setup class names
class_names = ['pizza', 'steak', 'sushi']

# Setup device
device = "cuda" if torch.cuda.is_available() else 'cpu'


# Get the image path
IMG_PATH = args.image_path
print(f"[INFO] Predicting on {IMG_PATH}")


def load_model(model_path: str=args.model_path):
    '''Load the saved model state dictionary
    Args:
        filepath: the filepath of the saved model
    '''
    # Need to use same hyperparameters as asaved model
    model = model_builder.TinyVGG(input_shape=3, 
                                  hidden_units=64, 
                                  output_shape=3).to(device)

    print(f"[INFO] Loading model from: {model_path}")

    # Load the model state_dict()
    model.load_state_dict(torch.load(f=model_path))

    return model


def prediction_on_image(image_path=IMG_PATH, model_path=args.model_path):
    # Load the model
    model = load_model(model_path=model_path)

    # Load the image and turn it into torch.float32 (same type as model)
    image = torchvision.io.decode_image(str(image_path)).type(torch.float32)

    # Divide the image pixel values by 255 to get them between [0, 1]
    image = image / 255.

    # Transform the image
    transform = torchvision.transforms.Compose([torchvision.transforms.Resize(size=(64,64))])
    image = transform(image)

    # Predict on image
    model.eval()
    with torch.inference_mode():
        # Put the image into target device
        image = image.to(device)

        # Get pred logits
        pred_logits = model(image.unsqueeze(dim=0)) # make sure image has batch dimension (shape: [batch_size, color_channels, height, width])

        # Get pred probs and label
        pred_prob = torch.softmax(pred_logits, dim=1)
        pred_label = torch.argmax(pred_prob, dim=1)

        # Get the preb label class
        pred_label_class = class_names[pred_label]

    print(f"[INFO] Predicted Class: {pred_label_class} | Predicted class probability: {pred_prob.max():.3f}")


if __name__ == "__main__":
    prediction_on_image()






This folder contains an example submission for track 1 of the 2024 CVPR VAND Workshop Challenge.

# Evaluation Script

The submissions for track 1 of the challenge will be ranked by evaluating the models on all 15 categories of the MVTec dataset. The F1Max score (highest F1 score that can be achieved based on the raw anomaly score predictions) will be computed for the test set of each of the categories after applying a set of random transformations to the data to simulate domain drift.

The `evaluation.py` provided in the example submission shows an example of how we will run your model. Please ensure that your model is compatible with this evaluation script before sending your submission to the organizers. The evaluation script takes 3 command line arguments:

- `--model_path`: This is the class path to the python module that contains your model class.
- `--model_name`: This is the class name of your main model class.
- `--weights_path`: Path to the folder which contains the weigth files. The weight folder should contain a single weight file for each of the 15 MVTec categories.

For example, if your model is defined as `MyModel` in `my_team_name/my_model.py`, and your weight files are stored in the `my_weights_folder` folder, you could run the evaluation script with the following command:

```python
python evaluation.py \
        --model_path <my_team_name.my_model> \
        --model_name <MyModel> \
        --weights_path <my_weights_folder>
        --category <dataset_category>
```

# Model requirements

All models should be implemented in PyTorch. The model definition should be provided in a Python file, and the model itself should be a Python class which inherits from `torch.nn.Module`. The model should implement the `forward` method which receives the input images and returns the image- and/or pixel-level predictions. All interaction of the evaluation script with your model will be through this`forward` method.

## Inputs

The evaluation script will resize the input images to a width and height of `256x256`. This is to minimize the running time and memory use of the models and to ensure that no advantage can be gained by processing the images at a higher resolution. Any additional data transforms that your model may need should be part of the `forward` method. The images are loaded with a batch size of 1, so each input image passed to the `forward` method is a float Tensor with pixel values between 0 and 1 and with a shape of `[1, 3, 256, 256]`. The data type is `torch.float32` and the channel order is `RGB`.

## Outputs

The predictions of your model should be returned as a dictionary. The only required key is `pred_score`, which should contain the predicted anomaly score for the given image. The value of `pred_score` should be a float tensor of shape `[1]`. If your model produces pixel-level anomaly score predictions, you can add these to the dictionary under the `anomaly_maps` key. The anomaly maps should be a float tensor of shape `[1, 256, 256]`. If your model generates pixel-level predictions, we will compute and report the pixel-level F1 Max score of your model in addition to the image-level scores, but please be aware that only the image-level predictions will be used for the final ranking of the submissions.

# Submission Instructions

Please read these instructions for submission carefully to ensure that nothing is missing from your submission.

Your submission should consist of the following components:

- A python module which contains the implementation of your model. The model implementation should contain all the necessary parts needed to evaluate your model on the tests set of the challenge dataset. The module may consist of a single Python file or a folder of Python files.

- A requirements.txt file, which lists all the package dependencies needed to run your model. An easy way to obtain the requirements file is to use `pip freeze > requirements.txt`. You are free to use any python packages needed.

- A folder of weight files, where each weight file is named after the dataset category on which the model was trained. It is expected that a weights file is provided for each of the 15 MVTec categories, for example:

    ```bash
    my_weights
    ├── bottle.pt
    ├── cable.pt
    ├── ...
    ├── wood.pt
    └── zipper.pt
    ```

- A README.md file with instructions how to set up the environment and which values to pass to the evaluation script as command line arguments.

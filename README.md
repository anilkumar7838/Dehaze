# Dehazing Image App

This project is a Streamlit application for dehazing images using a deep learning model. The app allows users to upload a hazy image, process it using a pre-trained model, and download the dehazed image.

## Features

- Upload hazy images in `.png`, `.jpg`, or `.jpeg` formats.
- Process the uploaded image to remove haze using a deep learning model.
- Download the dehazed image.

## Prerequisites

- Python 3.8 or later.
- A compatible GPU (optional but recommended for faster processing).

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

3. Activate the virtual environment:

   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

4. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

5. Update the paths for the model and input/output folders in the `app.py` file.

## Running the App

1. Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```

2. Open the app in your browser:
   - Local: `http://localhost:8501`
   - Network: Use the displayed network URL.

## Docker Instructions

1. Build the Docker image:
   ```bash
   docker build -t dehaze-app .
   ```

2. Run the container:
   ```bash
   docker run -p 8501:8501 dehaze-app
   ```

## File Structure

- `app.py`: Main Streamlit application.
- `model.py`: Model loading and initialization.
- `runtestModel.py`: Dehazing logic.
- `requirements.txt`: Python dependencies.
- `Dockerfile`: Docker image configuration.
- `README.md`: Project documentation.

## Notes

- Ensure that the `MODEL_PATH` and folder paths in `app.py` are correctly set before running the app.
- GPU support requires PyTorch installed with CUDA.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

Special thanks to the authors of the dehazing model and the open-source community for tools and libraries.

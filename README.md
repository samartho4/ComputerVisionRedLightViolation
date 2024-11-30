Create a Virtual Environment
Using venv:

python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
Using conda:

conda create -n redlight_env python=3.8
conda activate redlight_env
Install Dependencies
pip install -r requirements.txt
Usage

Running the Detection Script
python main.py
Command-Line Arguments (Optional)
--input: Path to the input video file. Defaults to traffic_video.mp4.
--output: Path to save the output video with detections. Defaults to output_video.mp4.
Example:

python main.py --input videos/traffic_video.mp4 --output outputs/detected_video.mp4
Output
The script will display the video with detected violations in real-time.
Violations are logged in logs/violations.log with timestamps and details.
Project Structure

ComputerVisionRedLightViolation/
├── config.py
├── database.py
├── license_plate.py
├── main.py
├── requirements.txt
├── stop_line_detection.py
├── traffic_light.py
├── utils.py
├── videos/
│   ├── traffic_video.mp4
│   └── ...
├── outputs/
│   └── detected_video.mp4
├── logs/
│   └── violations.log
├── models/
│   └── yolov5/
├── docs/
│   ├── Project_Report.pdf
│   └── images/
│       ├── banner.png
│       └── demo_thumbnail.png
└── README.md
Contributing

Contributions are welcome! Please follow these steps:

Fork the repository
Create a new branch for your feature or bugfix:
git checkout -b feature/your-feature-name
Commit your changes:
git commit -am 'Add a new feature'
Push to the branch:
git push origin feature/your-feature-name
Open a Pull Request
Please ensure your code adheres to the project's coding standards and includes appropriate tests.

License

This project is licensed under the MIT License.

Contact

Email: samarthx04@gmail.com

Acknowledgments

Farzad Nekouei - Original Kaggle notebook inspiration.
Ultralytics - YOLOv5 repository.
Open-source libraries: OpenCV, TensorFlow/PyTorch.
References

Project Report

YOLOv5 Documentation
OpenCV Documentation

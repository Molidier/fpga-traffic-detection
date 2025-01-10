# Vehicle Tracking and Counting System  

This project implements a real-time vehicle tracking and counting system with external hardware integration for live data transmission. It features accurate detection, smooth tracking, and dynamic updates for vehicle count monitoring.  

---

## Features  

- **Vehicle Detection and Tracking:**  
  Detects and tracks vehicles in real-time using video feeds with object detection and position smoothing.  

- **Counting and Reporting:**  
  Accurately counts vehicles passing a defined line and transmits the count to external hardware via UART.  

- **Histogram-Based Tracking:**  
  Uses histograms for appearance matching to improve tracking consistency.  

---

## Project Structure  

| File/Folder        | Description                                                                 |
|--------------------|-----------------------------------------------------------------------------|
| `fpga_traffic.py`  | Main script for vehicle detection, tracking, counting, and FPGA communication. |
| `data/`            | Contains pre-trained models, video inputs, and related resources.           |
| `sort/`            | SORT tracking algorithm implementation for real-time object tracking.       |
| `output/`          | Stores processed videos and logs (can remain empty initially).              |
| `src/`             | Folder for additional source code and utilities.                           |

---

## Prerequisites  

- **Python 3.8+**  
- Required packages:  
  - `numpy`  
  - `opencv-python`  
  - `pyserial`  
  - `matplotlib` (optional for visualization)  

---

## Installation  

1. Clone the repository:  

   ```bash
   git clone https://github.com/your_username/vehicle_tracking_system
   cd vehicle_tracking_system
   ```

2. Create and activate a virtual environment:  

   ```bash
   python -m venv venv  
   source venv/bin/activate  # On Windows: venv\Scripts\activate  
   ```

3. Install dependencies:  

   ```bash
   pip install -r requirements.txt  
   ```

---

## Usage  

1. Prepare your input:  
   - Place your video file in the `data/` directory.  
   - Ensure the required cascade file (e.g., `cars.xml`) is present in the `data/` folder.  

2. Run the main script:  

   ```bash
   python fpga_traffic.py
   ```

3. View the real-time tracking and counting output on your screen.  

4. Vehicle count is transmitted to the external FPGA hardware via UART.  

---

## System Details  

- **Detection:**  
  Utilizes pre-trained cascade classifiers to identify vehicles.  

- **Tracking:**  
  Employs tracking with smoothing for stable vehicle tracking over time.  

- **Counting Logic:**  
  Detects when vehicles cross a designated line and increments the count dynamically.  

- **Data Transmission:**  
  Sends vehicle count to external hardware in real-time over a UART connection.  

---

## Author  
Developed by Moldir Azhimukhanbet.  

--- 

Let me know if you'd like additional modifications!

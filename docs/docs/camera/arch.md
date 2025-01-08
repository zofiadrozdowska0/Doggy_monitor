# **Architektura Systemu**

Projekt “Doggy Monitor” oparty jest na Raspberry Pi Zero, który zarządza kamerą oraz serwami obsługującymi gimbal. Cały system jest zaprojektowany, aby monitorować psa i przesyłać obraz do chmury w czasie rzeczywistym. Algorytmy do śledzenia ruchu psa umożliwiają automatyczne sterowanie serwami gimbala, aby utrzymać psa w kadrze.

## **Diagramy**

<!-- ### Klasy

```mermaid
classDiagram
    Tracker <|-- PIDController
    Tracker <|-- ServoController
    Tracker <|-- KalmanFilter

    class Tracker {
        -host: str
        -port: int
        -frame_width: int
        -frame_height: int
        -frame_center_x: int
        -frame_center_y: int
        -pid_x: PIDController
        -pid_y: PIDController
        -servo_controller: ServoController
        -kalman: KalmanFilter
        -detections_in_last_2_sec: list
        +start_camera()
        +stream_frames()
        +update_kalman_transition_matrix(float dt)
        +watchdog()
        +create_kalman_filter()
        +manage_feedback_connections()
        +handle_client(socket client_socket)
        +control_servos(int pred_x, int pred_y)
        +stop()
    }

    class PIDController {
        -K_p: float
        -K_i: float
        -K_d: float
        -previous_error: float
        -integral: float
        +compute(float setpoint, float measured_value, float dt, float dead_zone=None)
        +reset()
    }

    class ServoController {
        -servo_pin_x: int
        -servo_pin_y: int
        -angle_file: str
        -current_angle_x: float
        -current_angle_y: float
        +move_x(float target_angle_x)
        +move_y(float target_angle_y)
        +move(float target_angle_x, float target_angle_y)
        +set_angle(int servo_pin, float angle)
        +save_angles_to_file(float angle_x, float angle_y)
        +read_angles_from_file()
        +stop()
    }

    class KalmanFilter {
        -measurementMatrix: numpy.ndarray
        -transitionMatrix: numpy.ndarray
        -processNoiseCov: numpy.ndarray
        -measurementNoiseCov: numpy.ndarray
        +predict()
        +correct(numpy.ndarray measurement)
    }
``` -->

### **Przepływ danych**

```mermaid
flowchart LR
    RPI-- image -->http
    http-- image -->server
    server-- emotion -->mobile_app
    http-- image -->mobile_app
    server-- tracking and cropping -->RPI
    RPI-- move to track dog -->move_servo
```

## **Zastosowane technologie**

System działa na Raspberry Pi Zero, który jest odpowiedzialny za obsługę zarówno serw, jak i kamery. Do obsługi pinów oraz serwa wykorzystywane są biblioteki Python:

• **RPi.GPIO** – do zarządzania pinami GPIO na Raspberry Pi.

• **pigpio** – do precyzyjnego sterowania sygnałami PWM dla serw.

• **picamera2** – do obsługi kamery Raspberry Pi i przesyłania obrazu.

**Struktura plików projektu**

```mermaid
graph TD
    src --> main.py
    src --> servo_controller.py
```

• `main.py`: Obsługa strumieniowania obrazu z kamery i śledzenia psa.

• `servo_controller.py`: Kod odpowiedzialny za sterowanie serwami.
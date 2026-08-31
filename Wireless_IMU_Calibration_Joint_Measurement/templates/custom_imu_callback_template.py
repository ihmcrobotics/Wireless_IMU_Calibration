"""
Template callback adapter for custom IMU integration with ReBAIT

This template provides a starting point for integrating your IMU.
Replace the TODO sections with your IMU's specific API calls.

Author: ReBAIT Team
Modified for: [Your IMU Model]
"""

from threading import Lock, Thread
from warnings import warn
import time
import numpy as np

# TODO: Replace with your IMU's SDK import
# from your_imu_sdk import YourIMUAPI, YourIMUPacket


class CustomIMUCallback:
    """
    Adapter class for custom IMU that implements the ReBAIT callback interface.
    
    This class handles communication with your IMU and converts its data format
    to the standard ReBAIT format expected by DataCollector.
    
    Required interface:
    - devIds: List of device IDs
    - samp_freq: Sampling frequency (Hz)
    - packetAvailable(): Check if data ready
    - getNextPacket(): Retrieve and parse next packet
    - enable(): Start IMU data collection
    - attach(): Start callback/streaming
    - detach(): Stop callback/streaming
    - close(): Cleanup and disconnect
    """
    
    def __init__(self, max_buffer_size=5, samp_freq=100, port=None):
        """
        Initialize IMU connection and configuration.
        
        Args:
            max_buffer_size (int): Size of packet buffer (increase if dropping packets)
            samp_freq (int): Target sampling frequency in Hz (must match IMU config)
            port (str): COM port or device path (e.g., 'COM3', '/dev/ttyUSB0')
        """
        self.samp_freq = samp_freq
        self.port = port
        self.m_maxNumberOfPacketsInBuffer = max_buffer_size
        self.m_packetBuffer = []
        self.m_lock = Lock()
        self.attached = False
        self.read_thread = None
        
        # TODO: Initialize your IMU hardware here
        # Example for serial-based IMU:
        # import serial
        # self.serial = serial.Serial(port, baudrate=115200, timeout=1)
        # time.sleep(1)  # Wait for connection
        
        # Get connected devices
        self.devIds = self._discover_devices()
        
        if len(self.devIds) == 0:
            raise RuntimeError('No IMU devices found. Check connection and port.')
    
    def _discover_devices(self):
        """
        Discover and list all connected IMU devices.
        
        TODO: Implement for your IMU
        Should return list of device ID strings, e.g., ['IMU_001', 'IMU_002']
        
        Returns:
            list: Device IDs
        """
        # Placeholder - replace with actual device discovery
        device_ids = ['IMU_001']  # TODO: Get actual device IDs from IMU
        print(f'Found {len(device_ids)} device(s): {device_ids}')
        return device_ids
    
    def enable(self):
        """
        Enable IMU and start measurement mode.
        
        This is called before data collection begins.
        Typically involves:
        - Enabling radio/wireless connection
        - Switching to measurement mode
        - Configuring data output rate
        """
        try:
            # TODO: Implement for your IMU
            # Example:
            # self.imu.enable_radio()
            # self.imu.goto_measurement_mode()
            # self.imu.set_update_rate(self.samp_freq)
            print(f'IMU enabled at {self.samp_freq} Hz')
        except Exception as e:
            raise RuntimeError(f'Failed to enable IMU: {e}')
    
    def attach(self):
        """
        Attach callback to start receiving data packets.
        
        For event-driven IMUs: register your callback with the IMU
        For polling-based IMUs: start a background thread to read data
        """
        self.attached = True
        
        # TODO: Choose implementation based on your IMU type:
        
        # Option 1: Event-driven (Xsens, many wireless systems)
        # self.imu.add_callback_handler(self.on_data_received)
        
        # Option 2: Polling-based (USB serial, etc.)
        self.read_thread = Thread(target=self._polling_loop, daemon=True)
        self.read_thread.start()
    
    def detach(self):
        """
        Detach callback to stop receiving packets.
        """
        self.attached = False
        if self.read_thread:
            self.read_thread.join(timeout=1)
    
    def _polling_loop(self):
        """
        Background thread for polling-based IMUs.
        
        TODO: Implement this if your IMU doesn't provide event-driven callbacks
        """
        while self.attached:
            try:
                # TODO: Read from your IMU
                # Example for serial-based IMU:
                # if self.serial.in_waiting > 0:
                #     packet_data = self.serial.read(64)  # Adjust size
                #     packet = self._parse_packet(packet_data)
                #     self.m_lock.acquire()
                #     if len(self.m_packetBuffer) >= self.m_maxNumberOfPacketsInBuffer:
                #         self.m_packetBuffer.pop(0)
                #         warn('Buffer overflow. Dropping oldest packet.')
                #     self.m_packetBuffer.append(packet)
                #     self.m_lock.release()
                
                time.sleep(1.0 / self.samp_freq)  # Poll at sampling frequency
            except Exception as e:
                warn(f'Error in polling loop: {e}')
    
    def on_data_received(self, packet):
        """
        Callback for event-driven IMUs.
        
        TODO: Use this if your IMU provides event-driven callbacks
        This method is called automatically by the IMU when data is available.
        """
        self.m_lock.acquire()
        try:
            while len(self.m_packetBuffer) >= self.m_maxNumberOfPacketsInBuffer:
                self.m_packetBuffer.pop(0)
                warn('Buffer overflow. Dropping oldest packet.')
            self.m_packetBuffer.append(packet)
        finally:
            self.m_lock.release()
    
    def packetAvailable(self):
        """
        Check if a data packet is available in the buffer.
        
        Returns:
            bool: True if packet is available, False otherwise
        """
        self.m_lock.acquire()
        res = len(self.m_packetBuffer) > 0
        self.m_lock.release()
        return res
    
    def getNextPacket(self):
        """
        Retrieve the next available packet from buffer.
        
        Returns:
            tuple: (device_id: str, data: list) where data is
                   [timestamp, acc, gyr, quat, cal_acc, euler]
                   or (None, None) if buffer is empty
        """
        self.m_lock.acquire()
        if len(self.m_packetBuffer) == 0:
            self.m_lock.release()
            return None, None
        
        packet = self.m_packetBuffer.pop(0)
        self.m_lock.release()
        
        return self.packetExtract(packet)
    
    def packetExtract(self, packet):
        """
        Extract and convert IMU packet to standard ReBAIT data format.
        
        TODO: Adapt this to your IMU's packet structure
        
        The standard format is:
        data = [
            timestamp,           # float: seconds (Unix time or relative)
            acc,                # [x, y, z]: Free acceleration (m/s²)
            gyr,                # [x, y, z]: Calibrated gyroscope (rad/s)
            quat,               # [w, x, y, z]: Orientation quaternion
            cal_acc,            # [x, y, z]: Calibrated acceleration (m/s²)
            euler               # [pitch, yaw, roll]: Euler angles
        ]
        
        Args:
            packet: Raw packet from IMU (format depends on your IMU)
        
        Returns:
            tuple: (device_id: str, data: list) in standard format
        """
        try:
            # TODO: Replace with your IMU's packet structure
            # This is a placeholder example
            
            device_id = getattr(packet, 'device_id', self.devIds[0])
            
            # Extract timestamp (must be float in seconds)
            timestamp = packet.timestamp
            if not isinstance(timestamp, float):
                timestamp = float(timestamp)
            
            # Extract acceleration [x, y, z] in m/s²
            acc = np.array(packet.acceleration)
            
            # Extract gyroscope [x, y, z] in rad/s
            gyr = np.array(packet.gyroscope)
            
            # Extract quaternion [w, x, y, z]
            # NOTE: Some IMUs provide [x, y, z, w] - adjust if needed
            quat = np.array(packet.quaternion)
            if len(quat) == 4 and abs(quat[0]) <= 1 and abs(quat[0]) >= 0.1:
                # If first component looks like 'w', assume correct order
                pass
            else:
                # If first component looks like 'x', swap to [w, x, y, z]
                quat = np.array([quat[3], quat[0], quat[1], quat[2]])
            
            # Extract calibrated acceleration [x, y, z] in m/s²
            # If not available, use same as free acceleration
            cal_acc = np.array(packet.calibrated_acceleration)
            if cal_acc is None:
                cal_acc = acc
            
            # Extract Euler angles [pitch, yaw, roll] in radians
            # If in degrees, convert: radians = degrees * pi / 180
            euler = np.array(packet.euler_angles)
            if euler[0] > np.pi or euler[1] > np.pi or euler[2] > np.pi:
                # Probably in degrees, convert to radians
                euler = euler * np.pi / 180.0
            
            data = [timestamp, acc, gyr, quat, cal_acc, list(euler)]
            
            return device_id, data
            
        except Exception as e:
            warn(f'Error extracting packet: {e}')
            return None, None
    
    def close(self):
        """
        Close IMU connection and cleanup resources.
        
        This is called when data collection ends.
        Ensures proper shutdown sequence.
        """
        try:
            self.attached = False
            
            # TODO: Implement cleanup for your IMU
            # Example:
            # self.imu.close()
            # if hasattr(self, 'serial'):
            #     self.serial.close()
            
            print('IMU connection closed')
        except Exception as e:
            warn(f'Error closing IMU: {e}')


# ============================================================================
# HELPER FUNCTIONS - Uncomment and use as needed for your IMU
# ============================================================================

def rotation_matrix_to_quaternion(R):
    """
    Convert 3x3 rotation matrix to quaternion [w, x, y, z].
    
    Use this if your IMU provides orientation as a rotation matrix.
    
    Args:
        R: 3x3 rotation matrix (numpy array)
    
    Returns:
        np.array: Quaternion [w, x, y, z]
    """
    trace = np.trace(R)
    
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    
    return np.array([w, x, y, z])


def euler_to_quaternion(roll, pitch, yaw):
    """
    Convert Euler angles to quaternion [w, x, y, z].
    
    Use this if your IMU provides orientation as Euler angles (roll, pitch, yaw)
    but you need to convert to quaternion format.
    
    Args:
        roll, pitch, yaw: Euler angles in radians
    
    Returns:
        np.array: Quaternion [w, x, y, z]
    """
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    
    return np.array([w, x, y, z])


def quaternion_to_euler(quat):
    """
    Convert quaternion [w, x, y, z] to Euler angles [roll, pitch, yaw] in radians.
    
    Use this if your IMU provides quaternion but you need Euler angles.
    
    Args:
        quat: Quaternion [w, x, y, z]
    
    Returns:
        np.array: Euler angles [roll, pitch, yaw] in radians
    """
    w, x, y, z = quat
    
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)
    else:
        pitch = np.arcsin(sinp)
    
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    return np.array([roll, pitch, yaw])

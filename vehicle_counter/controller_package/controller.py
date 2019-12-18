from time import strftime as get_time
from datetime import date, datetime
import csv
import cv2

from tracker.sort import Sort as Tracker
from detector.tensorflow_mobilenet_ssd_v1 import VehicleDetector as Detector
from controller_package.interface import VehicleCounter

class Controller(VehicleCounter):
    
    def __init__(self, total_passed_vehicle = 0, ids_met_roi = [], write_now = False, ROI_POSITION=200):
        super().__init__()
        # self.__observers = observers
        self.__total_passed_vehicle = total_passed_vehicle #private
        self.__ids_met_roi = ids_met_roi
        # self.__write_now = write_now
        self.__ROI_POSITION = ROI_POSITION
    
    def check_logic(self, trackers):
        for track in trackers:
            if (track[3] > self.__ROI_POSITION) and (int(track[4]) not in self.__ids_met_roi): #(int(track[4]) != int(last_id_met_roi_line)):
                self.__total_passed_vehicle += 1 
                # last_id_met_roi_line = int(track[4])
                self.__ids_met_roi.append(int(track[4]))

    def count_on_frame(self, frame, detector):
        
        tracker = Tracker(max_age=25, min_hits=3)
        detections = detector.get_detections(frame)
        trackers = tracker.get_trackers(detections)
        self.check_logic(trackers)

    def check_time(self, **kwargs):
        #check if 1 hour passed
        write_now = (int( get_time('%M') ) == 0) and (int( get_time('%S') ) == 0)
        #check for another conditions 
        for arg in kwargs.values():
            write_now = bool(write_now + arg)
        return write_now
    
    def control_result_writer(self, **kwargs):
        # cam_name = kwargs['cam_name'] if 'cam_name' in kwargs.keys() else 'unknown'
        if 'cam_name' in kwargs.keys():
            cam_name = kwargs['cam_name']
            del kwargs['cam_name']
        else:
            cam_name = 'unknown'

        if self.check_time(**kwargs):
            with open('/media/trung/01D4B61EC8BD72C0/Ki1_nam5/HHTQĐ/TRUNG_vehicle_counting_project/traffic_measurement.csv', 'a') as f:
                writer_csv = csv.writer(f)                
                today = date.today().strftime('%d/%m/%Y')
                start_time = int(datetime.now().strftime('%H'))
                end_time = start_time + 1
                csv_line = '{},{},{},{},{}'.format(cam_name, today,start_time, end_time, self.__total_passed_vehicle)
                writer_csv.writerows([csv_line.split(',')])
        

if __name__ == '__main__':

    controller = Controller()
    detector = Detector()
    frame = cv2.imread('/media/trung/01D4B61EC8BD72C0/Ki1_nam5/HHTQĐ/TRUNG_vehicle_counting_project/vehicle_counter/test_data/test_image.jpeg')
    controller.count_on_frame(frame, detector)
    controller.control_result_writer(cam_name = 'cam1', write_now = True)
    print('DONE')
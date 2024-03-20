import cv2
import numpy as np

def detect_and_track_bowling_ball(video_path):
    # 비디오 파일 열기
    cap = cv2.VideoCapture(video_path)

    # 비디오 파일이 정상적으로 열렸는지 확인
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # 첫 번째 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        return

    frame_count = 0
    while True:
        frame_count += 1

        # 그레이스케일 변환
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 가우시안 블러 적용하여 잡음 제거
        blurred = cv2.GaussianBlur(gray, (11, 11), 0)

        # Hough Circle Transform을 사용하여 원 검출
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=30,
                                   param1=50, param2=30, minRadius=10, maxRadius=30)

        if circles is not None:
            # 원 검출되면 추적 시작
            circles = np.round(circles[0, :]).astype("int")
            (x, y, r) = circles[0]  # 첫 번째 원에 대해서만 추적

            # 초기 추적 위치 설정
            bbox = (x - r, y - r, 2 * r, 2 * r)
            tracker = cv2.TrackerMIL_create()
            tracker.init(frame, bbox)

            # 검출된 원 그리기
            cv2.circle(frame, (x, y), r, (0, 255, 0), 1)
            cv2.circle(frame, (x, y), 2, (0, 0, 255), 1)
            cv2.putText(frame, f"Radius: {r}px", (x - r, y - r - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            break  # 원 검출 후 반복문 종료

        # 다음 프레임 읽기
        ret, frame = cap.read()
        if not ret:
            break

    if circles is not None:
        # 비디오 캡처를 계속하여 볼링공 추적
        while True:
            # 새 프레임 읽기
            ret, frame = cap.read()
            if not ret:
                break

            # 추적
            success, bbox = tracker.update(frame)
            if success:
                # 추적 결과 시각화
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame, p1, p2, (255, 0, 0), 2)

                # 사각형 내에 반지름 정보 다시 계산
                rect_width = int(bbox[2])
                rect_height = int(bbox[3])
                new_radius = min(rect_width, rect_height) // 2

                # 추적된 볼링공의 크기 표시
                cv2.putText(frame, f"Bowling ball", (p1[0], p1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # 프레임에 현재 시간 표시
                cv2.putText(frame, f"Frame: {frame_count}", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # 결과 프레임 표시
                cv2.imshow("Tracking", frame)

                # 종료 조건
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                cv2.putText(frame, "Lost", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                cv2.imshow("Tracking", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        # 비디오 캡처 객체 닫기
        cap.release()
        cv2.destroyAllWindows()

    else:
        print("No bowling balls detected.")

# 동영상 파일 경로
video_path = "./video3.mp4"
# 볼링공 검출 및 추적 함수 호출
detect_and_track_bowling_ball(video_path)

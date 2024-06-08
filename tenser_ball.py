import numpy as np
import matplotlib.pyplot as plt

# 초기 상태와 공분산 설정
x_hat = np.array([0, 0])  # 초기 예측 상태 [x, x_dot]
P = np.eye(2)             # 초기 예측 공분산
Q = np.eye(2) * 0.01      # 프로세스 노이즈 공분산
R = np.array([[0.1]])     # 측정 노이즈 공분산

# 상태 및 측정 행렬 정의
A = np.array([[1, 1], [0, 1]])  # 상태 전이 행렬
H = np.array([[1, 0]])          # 측정 행렬

# 측정값 및 시간 설정
measurements = np.random.normal(loc=0, scale=0.1, size=100)  # 가상의 측정값
dt = 1  # 시간 간격

# 결과 저장을 위한 리스트 생성
x_hat_list = []
x_list = []

# 예측 및 업데이트 단계 반복
for z in measurements:
    # 예측 단계
    x_hat_minus = np.dot(A, x_hat)
    P_minus = np.dot(A, np.dot(P, A.T)) + Q

    # 칼만 이득 계산
    K = np.dot(P_minus, H.T) / (np.dot(H, np.dot(P_minus, H.T)) + R)

    # 업데이트 단계
    x_hat = x_hat_minus + np.dot(K, (z - np.dot(H, x_hat_minus)))
    P = np.dot((np.eye(2) - np.dot(K, H)), P_minus)

    # 결과 저장
    x_hat_list.append(x_hat[0])
    x_list.append(x_hat_minus[0])

# 차트 그리기
plt.figure(figsize=(10, 5))
plt.plot(measurements, label='Measurements', color='blue', marker='o')
plt.plot(x_hat_list, label='Estimated State', color='red')
plt.plot(x_list, label='Predicted State', color='green', linestyle='--')
plt.xlabel('Time')
plt.ylabel('Position')
plt.title('Extended Kalman Filter: Position Estimation')
plt.legend()
plt.grid(True)
plt.show()

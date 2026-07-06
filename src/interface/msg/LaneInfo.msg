# 차선 인지 결과 (차선 정중앙 주행용)
std_msgs/Header header

# 횡방향 오차: 이미지 중심 대비 차선 중심의 정규화 위치 [-1.0 ~ 1.0]
# 양수(+) = 차선 중심이 이미지 중심보다 오른쪽  → 차량이 왼쪽으로 치우침 → 우조향 필요
# 음수(-) = 차선 중심이 이미지 중심보다 왼쪽    → 차량이 오른쪽으로 치우침 → 좌조향 필요
float32 lane_offset

bool left_detected       # 좌측 차선 검출 여부
bool right_detected      # 우측 차선 검출 여부
float32 confidence       # 인식 신뢰도 0.0~1.0

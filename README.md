# AlexNet_prac

# To Do
오버플로우 처리하기(클리핑걸기)
연산 맞는지 확인 추가로하기(세부검증필요)

# 수정사항
int4*int4 하면 8bit int가 생길꺼고, accumulation 하는거 널널하게 주기위해서 16비트에 accumultion
layer 연산끝나고나서는 clipping 적용(int4 range)
실제 모델이라면 accuracy 감소로 생각해야할 점이 더 있겠지만, 이 경우는 굳이 필요없음 하드웨어 설계 감잡기 위해서 테스트용 코드 제작이니까

# How to Use
1. 받아간 다음에 SW 윗 경로에서 make하면 따라락 .exe 파일 생성됨
2. 생성된 alexnet.exe를 run하면 값들 슈루룩 나옴
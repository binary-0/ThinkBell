# 2021Woongjin_boys
##### This is the private repository of 2021 Woongjin Thinkbig Industry-Academic Cooperation Project Team.

## 통합 개발 상태
LocalHost에서 영상을 통한 측정(얼굴 각도, 눈 감김 측정) 및 음성을 통한 측정(전체 기간 중 발화율) 을 분석 가능하게 함.

## Trial and Error :
시도 1. JS기반 webRTC에 음성 및 영상 인식은 python으로 진행 시도
>> aioRTC(Python기반)을 활용해 음성 및 영상을 python에서 받아오고 python에서 처리 함 
>> 성공

시도 2. NIPA서버에 turn서버를 구축해 webRTC가 다른 IP에서 접속이 가능하도록 시도
>> 다수의 시도를 통해 turn서버 구축을 시도, 테스트 사이트를 활용한 브라우저 환경에서의 테스트는 성공, 그러나 configuration에 적용을 하면 응답이 오지 않음 
>> 실패, 추가적인 학습 및 멘토 필요

시도 3. aioRTC, webRTC를 활용해 실제 서비스를 제공하고 있는 Jitsi와 같은 플랫폼에 직접 적용 시도
>> 기능이 많아짐에 따라 이해도가 낮아져 기초적인 실행도 일부 실패 
>> 3~4일 정도의 추가적인 시도 뒤엔 향상된 결과 예상

## 향후 과제
1. 동일 IP가 아닌 다른 IP를 통한 접속을 가능하게 해야함 (Turn 서버 구현 필요)
2. CNN을 이용한 몰입도 측정
3. 음성 측정을 실시간을 확인 가능하도록 develop

# 2021Woongjin_boys
This is the private repository of 2021 Woongjin Thinkbig Industry-Academic Cooperation Project Team.

##개발 현황(2021.07.01)
####확실하게 완료된 사항: Local에서 영상을 통한 측정(얼굴 각도, 눈 감김 측정) 및 음성을 통한 측정(전체 기간 중 발화율) 가능

현재 리포지토리에 담겨 있는 코드는 크게 3가지의 모듈로 구성되어 있습니다.

*OpenCV를 이용한 영상 분석 모듈: Head Pose 측정, 얼굴 면적 측정, EAR(눈의 종횡비) 측정
*MFCC를 이용한 음성 분석 모듈: 발화 시간을 측정해 총 시간 대비 발화 시간의 비율 측정
*Javascript 기반 WebRTC 화상 회의 모듈

OpenCV와 MFCC는 Python을 이용해 작성되었고, WebRTC는 Javascript를 이용해 작성되었습니다.
OpenCV와 MFCC는 Flask를 이용해 Web App 환경에 이식 완료하였습니다.

본 코드는 두 언어 간의 통합을 시도하였고, Javascript에서 Python 프로세스를 생성하여 WebRTC 화상 회의를 하는 동시에 다른 프로세스로 Python 영상 처리를 진행함으로써 이를 해결하고자 하였습니다. 아래는 해당 프로세스 생성 코드입니다.

```javascript
const spawn = require('child_process').spawn; 
const result = spawn('python', ['Main.py']); 
result.stdout.on('data', function(data) { 
    console.log(data.toString()); 
}); 
result.stderr.on('data', function(data) { 
    console.log(data.toString()); 
});
```

##문제 상황(2021.07.01)
위와 같은 통합 방식으로 개발을 진행하였으나 여러 문제점이 발생하였습니다. 현재 리포지토리에 push되어 있는 코드의 문제 상황은 다음과 같습니다.

*단순히 두 가지 개별적 모듈을 동시에 구동하는 방식으로 Python Flask에서 따로, JS Express에서 따로 서버를 생성해서 같이 동작해야 할 모듈임에도 다른 포트를 이용해야 각자의 기능을 온전히 이용할 수 있었습니다.
*위의 문제와 마찬가지로, 개별적인 두 개의 모듈이 하나의 카메라 자원을 동시에 따로 이용할 수 없었습니다.
*이 문제를 해결하기 위해 WebRTC에서 얻는 웹캠 영상 자원을 프레임 단위로 Python에 전송시켜 이를 OpenCV에서 받아 처리하려고 했으나, JS 기반 WebRTC의 Stream 영상 처리 포맷과 OpenCV의 프레임 단위 영상 처리 포맷이 binding되기 어려웠습니다.

##Trial and Error (리포지토리에는 존재하지 않는 시도들)
문제 상황을 해결하기 위해 AioRTC를 채택하기로 했고, 아래는 그 과정에서 발생한 시도와 문제점입니다.

*시도 1. JS기반 webRTC에 음성 및 영상 인식은 python AioRTC로 진행 시도
    *AioRTC(Python기반)을 활용해 음성 및 영상을 python에서 받아오고 python에서 통신을 처리합니다.
    *로컬에서 혼자 접속할 때 영상 처리는 성공하였지만, Python 코드인 서버에서 영상 처리를 진행하였기 때문에 이 영상 처리를 Client에게 하도록 옮겨야 하지만 여전히 Javascript인 Client 코드에 이를 어떻게 이식해야 할 지 곤란합니다.

아래는 AioRTC의 server 파일에서 영상 처리 모듈을 클래스화하여 이식한 "FG"를 이용해 15프레임마다 영상 처리를 진행하는 코드입니다.
```python
self.frCtrl += 1

if self.frCtrl % 15 == 0:
    self.frCtrl = 0
    img = frame.to_ndarray(format="bgr24")
    img = self.FG.gen_frames(img)

    if img is not None:
        new_frame = VideoFrame.from_ndarray(img, format="bgr24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
        return new_frame
    else:
        return frame
else:
    return frame
```

*시도 2. NIPA서버에 turn서버를 구축해 webRTC가 다른 IP에서 접속이 가능하도록 시도
    *다수의 시도를 통해 turn서버 구축을 시도하였고, 테스트 사이트를 활용한 브라우저 환경에서의 테스트는 성공하였으나 configuration에 적용을 하면 응답이 오지 않습니다.
    *->실패, 추가적인 학습 및 멘토 필요

*시도 3. aioRTC, webRTC를 활용해 실제 서비스를 제공하고 있는 Jitsi와 같은 플랫폼에 직접 적용 시도
    *기능이 많아짐에 따라 이해도가 낮아져 기초적인 실행도 일부 실패하였습니다.
    *->3~4일 정도의 추가적인 시도 뒤엔 향상된 결과 예상

## 향후 과제
1. 동일 IP가 아닌 다른 IP를 통한 접속을 가능하게 해야함 (Turn 서버 구현 필요)
2. CNN을 이용한 몰입도 측정
3. 음성 측정을 실시간을 확인 가능하도록 develop
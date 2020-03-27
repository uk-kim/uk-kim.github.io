# GitHub 사용 방법
여기에서 기록할 GitHub 사용 방법에 관한 내용은 책 ["지옥에서 온 문서관리자, 깃&깃허브 입문"](http://www.kyobobook.co.kr/product/detailViewKor.laf?ejkGb=KOR&mallGb=KOR&barcode=9791163031222&orderClick=LAG&Kc=)을 보고 따라한 내용과
구글링을 통해 알게된 내용입니다.

## 00. Installation
### For Linux (Ubuntu)
```bash
sudo apt-get install git
```
### For Mac
```bash
sudo brew install git
```

## 01. Configuration
### 사용자 정보 설정
사용자 정보는 버전이 바뀔 때마다 함께 저장되기 때문에, 사용자 정보를 입력해야함.
`--global` 옵션을 추가함으로써 현재 컴퓨터의 모든 저장소에서 같은 사용자 정보를 사용한다.
```bash
$ git config --global user.name "Kimsu"
$ git config --global user.email "seongukzzz@gmail.com"
```

## 02. Manage version with Git
### 02-1. Make git repository
* 깃 초기화하기 : git init
1. 저장소를 만들 `hello-git` 디렉토리를 생성
```bash
$ mkdir hello-git
$ cd hello-git
```
###
2. 깃 초기화
```bash
$ git init
```

2단계까지 수행하면, hello-git 경로 내에 없었던 `.git/` 경로가 생성된다.

### 02-2. Make Version
#### 버전(version)이란
문서 작성을 하다 보면 아래와 같이 초안으로부터 수정을 거듭할 때마다 별도의 파일을 생성하여 보관하게 된다.
이렇게 각 단계별로 생성된 것처럼 구별된 것을 버전이라고 한다.

"초안.ppt" --> "수정.ppt" --> "수정2.ppt" --> ... --> "최종.ppt" --> "진짜최종.ppt" --> ...    (무한반복)

하지만 위와 같이 버전 관리를 하게 되면, 1) 누가 2) 어떤 부분을 3) 어떻게 바꾸었는지를 확인하기 어려워 진다.

#### 스테이지 / 커밋
- 작업트리 : 파일의 수정, 저장 등의 작업을 수행하는 디렉토리
- 스테이지 : 버전으로 만들 파일이 대기하는 곳. 10개 파일 중 4개만 버전으로 만드려면 4개 파일만 스테이지로 전달
- 저장소(repository) : 스테이지의 파일들을 버전으로 만들어 저장하는 곳

#### 예시
1) hello.txt 수정 후 저장 --> 작업 트리에 저장
  - 상태 보기
  ```bash
  $ git status
  ```
2) 수정된 hello.txt 파일 버전 관리 --> 스테이지에 보관
  - 스테이징
  ```bash
  $ git add hello.txt
  ```
3) 버전을 만들기 위해 '커밋(commit)' 명령 통해 저장소로 저장 --> 저장소(Repository)
  - 스테이지 파일 커밋
  ```bash
  $ git commit -m "Write message here. My first commit."
  ```
  - 결과 로그 보기
  ```bash
  $ git log
  ```

4) 스테이징과 커밋 동시에 처리하기
```bash
git commit -am "Commit message!"
```








import os
import subprocess
import platform

# 파일 존재 확인
if not os.path.exists('optimized.cpp'):
    raise FileNotFoundError("optimized.cpp 파일을 찾을 수 없습니다!")

# 아키텍처별 컴파일 옵션 설정
arch = platform.machine().lower()
if arch in ['arm64', 'aarch64']:
    march_flag = '-march=armv8.3-a'
elif arch in ['x86_64', 'amd64']:
    march_flag = '-march=native'
else:
    march_flag = ''  # 기본 설정

# 컴파일 명령어 구성
compile_cmd = f"clang++ -O3 -std=c++17 {march_flag} -o optimized optimized.cpp"

try:
    # 컴파일 실행
    result = subprocess.run(compile_cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"컴파일 실패: {result.stderr}")
        exit(1)
    
    # 실행 파일 권한 확인
    if os.path.exists('./optimized'):
        os.chmod('./optimized', 0o755)
        
        # 프로그램 실행
        run_result = subprocess.run('./optimized', capture_output=True, text=True)
        print(run_result.stdout)
        if run_result.stderr:
            print(f"실행 중 경고/에러: {run_result.stderr}")
    else:
        print("컴파일된 실행 파일을 찾을 수 없습니다!")
        
except Exception as e:
    print(f"예외 발생: {e}")

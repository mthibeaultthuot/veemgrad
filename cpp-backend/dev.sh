#!/bin/bash

GREEN='\033[0;32m'
BLUE='\033[1;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' 


BUILD_DIR="build/"

DEV_MODE=false
BUILD_MODE=false
EXECUTABLE="./backend"



for arg in "$@" 
do 
  echo $arg
  case $arg in 
    --dev)
      DEV_MODE=true
      ;;
    --build)
      BUILD_MODE=true;
      ;;
  esac
done


if  $DEV_MODE &&  $BUILD_MODE;
then
  echo "Cannot have build and dev flag at the same time"
  exit 1
fi


if ! $DEV_MODE && ! $BUILD_MODE;
then
  echo "No flags where insert running in dev mode..."
  DEV_MODE=true
fi

dir_exist() {
  if [ -d "$BUILD_DIR" ]
  then 
    return 0
  fi
  return 1
}


run() {
 cd $BUILD_DIR
 make

 echo -e "${GREEN}🏃‍♂️ Running ${EXECUTABLE}...${NC}"
 $EXECUTABLE

 cd ..

 return 0
}


build() {
  cd $BUILD_DIR

  echo -e "${GREEN}🔨 Building project...${NC}"
  cmake ..

  make

  cd ..

  return 0
}


if $DEV_MODE 
then
  if ! dir_exist; 
  then 
    mkdir build
    build
  fi
  
  run
fi

if $BUILD_MODE
then
  if  dir_exist;
  then 
    rm -rf build
  fi

  mkdir build


  build
fi

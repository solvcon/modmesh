#!/bin/bash

# Install Apple Metal. See: https://developer.apple.com/metal/cpp/

script_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

download_md5 () {
  local loc=$1
  local url=$2
  local md5hash=$3
  if [ $(uname) == Darwin ] ; then
    local md5="md5 -q"
  elif [ $(uname) == Linux ] ; then
    local md5=md5sum
  fi
  if [ ! -e $loc ] || [ $md5hash != `$md5 $loc | cut -d ' ' -f 1` ] ; then
    mkdir -p $(dirname $loc)
    rm -f $loc
    echo "Download from $url"
    curl -sSL -o $loc $url
  fi
  local md5hash_calc=`$md5 $loc | cut -d ' ' -f 1`
  if [ $md5hash != $md5hash_calc ] ; then
    echo "$(basename $loc) md5 hash $md5hash but got $md5hash_calc"
  else
    echo "$(basename $loc) md5 hash $md5hash confirmed"
  fi
}

filename=metal-cpp_macOS12_iOS15.zip

pushd ${script_root}

download_md5 \
  ${script_root}/archive/${filename} \
  https://developer.apple.com/metal/cpp/files/metal-cpp_macOS12_iOS15.zip \
  8faab6897ba1f62e87076f153e036a58

unzip ${script_root}/archive/${filename} -d install -x __MACOSX/\*

popd
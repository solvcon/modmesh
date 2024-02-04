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

if [ $(uname) == Darwin ] ; then
  ver=$(sw_vers -productVersion)
  if [ ${ver:0:2} == "12" ] ; then
    filename=metal-cpp_macOS12_iOS15.zip
    url=https://developer.apple.com/metal/cpp/files/metal-cpp_macOS12_iOS15.zip
    checksum=8faab6897ba1f62e87076f153e036a58
  elif [ ${ver:0:2} == "13" ] ; then
    filename=metal-cpp_macOS13_iOS16.zip
    url=https://developer.apple.com/metal/cpp/files/metal-cpp_macOS13.3_iOS16.4.zip
    checksum=771a496981fb79dbd11d8bb128f19158
  elif [ ${ver:0:2} == "14" ] ; then
    filename=metal-cpp_macOS14_iOS17.zip
    url=https://developer.apple.com/metal/cpp/files/metal-cpp_macOS14.2_iOS17.2.zip
    checksum=8ec6c894233c834f7c611c575be72315
  else
    echo "Unsupported macOS version $ver"
    exit 1
  fi
else
  echo "Unsupported OS $(uname)"
  exit 1
fi

pushd ${script_root}

download_md5 \
  ${script_root}/archive/${filename} \
  ${url} \
  ${checksum}

unzip ${script_root}/archive/${filename} -d install -x __MACOSX/\*

popd
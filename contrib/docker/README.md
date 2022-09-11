# Docker

## Build
```bash
$ docker build -t solvcon/modmesh:latest contrib/docker
```

## Run
```bash
$ docker run -it --rm -v $(pwd):/workspace solvcon/modmesh:latest
```
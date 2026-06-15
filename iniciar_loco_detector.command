#!/bin/bash

# Cambiar a la carpeta donde está este script
cd "$(dirname "$0")" || exit 1

IMAGE_ARCHIVE="loco-detector-v1.0.0-docker-images.tar.gz"
IMAGE_TAR="loco-detector-v1.0.0-docker-images.tar"
COMPOSE_FILE="docker-compose.release.yml"
BACKEND_URL="http://127.0.0.1:8011/api/health"
FRONTEND_URL="http://127.0.0.1:5178"

echo "=========================================="
echo "Iniciando LOCO Detector"
echo "=========================================="
echo ""

# Verificar Docker
if ! command -v docker >/dev/null 2>&1; then
  echo "ERROR: Docker no está instalado o no está disponible en PATH."
  echo "Instala Docker Desktop y vuelve a ejecutar este archivo."
  read -p "Presiona Enter para salir..."
  exit 1
fi

# Verificar que Docker Desktop esté iniciado
if ! docker info >/dev/null 2>&1; then
  echo "ERROR: Docker Desktop no está iniciado."
  echo "Abre Docker Desktop, espera que termine de iniciar y vuelve a ejecutar este archivo."
  read -p "Presiona Enter para salir..."
  exit 1
fi

# Verificar Docker Compose
if ! docker compose version >/dev/null 2>&1; then
  echo "ERROR: Docker Compose no está disponible."
  echo "Actualiza Docker Desktop y vuelve a ejecutar este archivo."
  read -p "Presiona Enter para salir..."
  exit 1
fi

# Verificar archivo comprimido
if [ ! -f "$IMAGE_ARCHIVE" ]; then
  echo "ERROR: No se encontró $IMAGE_ARCHIVE."
  echo "Este archivo debe estar en la misma carpeta que este .command."
  read -p "Presiona Enter para salir..."
  exit 1
fi

# Verificar docker-compose release
if [ ! -f "$COMPOSE_FILE" ]; then
  echo "ERROR: No se encontró $COMPOSE_FILE."
  echo "Este archivo debe estar en la misma carpeta que este .command."
  read -p "Presiona Enter para salir..."
  exit 1
fi

# Descomprimir imágenes si el .tar no existe
if [ ! -f "$IMAGE_TAR" ]; then
  echo "Descomprimiendo imágenes Docker..."
  tar -xzf "$IMAGE_ARCHIVE"

  if [ $? -ne 0 ]; then
    echo "ERROR: No se pudo descomprimir $IMAGE_ARCHIVE."
    read -p "Presiona Enter para salir..."
    exit 1
  fi
fi

# Verificar que el TAR exista
if [ ! -f "$IMAGE_TAR" ]; then
  echo "ERROR: No se encontró $IMAGE_TAR después de descomprimir."
  read -p "Presiona Enter para salir..."
  exit 1
fi

# Cargar imágenes Docker
echo "Cargando imágenes Docker..."
docker load -i "$IMAGE_TAR"

if [ $? -ne 0 ]; then
  echo "ERROR: Docker no pudo cargar las imágenes."
  read -p "Presiona Enter para salir..."
  exit 1
fi

# Detener contenedores anteriores
echo "Deteniendo contenedores anteriores si existen..."
docker compose -f "$COMPOSE_FILE" down

# Levantar LOCO Detector
echo "Levantando LOCO Detector..."
docker compose -f "$COMPOSE_FILE" up -d

if [ $? -ne 0 ]; then
  echo "ERROR: No se pudo iniciar LOCO Detector."
  echo "Puede que los puertos 8011 o 5178 estén ocupados."
  echo ""
  echo "Logs recientes:"
  docker compose -f "$COMPOSE_FILE" logs --tail 80
  read -p "Presiona Enter para salir..."
  exit 1
fi

# Esperar backend
echo "Esperando backend..."
attempts=0

while true; do
  attempts=$((attempts + 1))

  if curl --silent --fail "$BACKEND_URL" >/dev/null 2>&1; then
    break
  fi

  if [ "$attempts" -ge 60 ]; then
    echo ""
    echo "ERROR: El backend no respondió a tiempo."
    echo ""
    echo "Logs recientes:"
    docker compose -f "$COMPOSE_FILE" logs --tail 80
    read -p "Presiona Enter para salir..."
    exit 1
  fi

  sleep 2
done

echo ""
echo "=========================================="
echo "LOCO Detector listo."
echo "Frontend: $FRONTEND_URL"
echo "=========================================="
echo ""

open "$FRONTEND_URL"

read -p "Presiona Enter para salir..."
exit 0
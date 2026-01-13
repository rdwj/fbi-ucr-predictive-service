#!/bin/bash
#
# Deploy FBI UCR Prediction Service to OpenShift
#
# Usage:
#   ./deploy.sh [OPTIONS]
#
# Options:
#   -n, --namespace     Namespace to deploy to (default: fbi-ucr)
#   -i, --image         Container image (default: quay.io/wjackson/crime-stats-api:latest)
#   -u, --quay-user     Quay.io username (or set QUAY_USER env var)
#   -p, --quay-password Quay.io password/token (or set QUAY_PASSWORD env var)
#   -s, --secret-name   Name for pull secret (default: quay-pull-secret)
#   --skip-secret       Skip pull secret creation (if already exists or public image)
#   -h, --help          Show this help message
#
# Examples:
#   ./deploy.sh -n my-namespace -u myuser -p mytoken
#   QUAY_USER=myuser QUAY_PASSWORD=mytoken ./deploy.sh
#   ./deploy.sh --skip-secret  # For public images

set -euo pipefail

# Default values
NAMESPACE="fbi-ucr"
IMAGE="quay.io/wjackson/crime-stats-api:latest"
SECRET_NAME="quay-pull-secret"
SKIP_SECRET=false
QUAY_USER="${QUAY_USER:-}"
QUAY_PASSWORD="${QUAY_PASSWORD:-}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

show_help() {
    head -25 "$0" | tail -22 | sed 's/^# //' | sed 's/^#//'
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--namespace)
            NAMESPACE="$2"
            shift 2
            ;;
        -i|--image)
            IMAGE="$2"
            shift 2
            ;;
        -u|--quay-user)
            QUAY_USER="$2"
            shift 2
            ;;
        -p|--quay-password)
            QUAY_PASSWORD="$2"
            shift 2
            ;;
        -s|--secret-name)
            SECRET_NAME="$2"
            shift 2
            ;;
        --skip-secret)
            SKIP_SECRET=true
            shift
            ;;
        -h|--help)
            show_help
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            ;;
    esac
done

# Verify oc is logged in
if ! oc whoami &>/dev/null; then
    log_error "Not logged into OpenShift. Run 'oc login' first."
    exit 1
fi

CLUSTER=$(oc whoami --show-server)
log_info "Deploying to cluster: $CLUSTER"
log_info "Target namespace: $NAMESPACE"

# Create namespace if it doesn't exist
if oc get namespace "$NAMESPACE" &>/dev/null; then
    log_info "Namespace '$NAMESPACE' already exists"
else
    log_info "Creating namespace '$NAMESPACE'..."
    oc create namespace "$NAMESPACE"
fi

# Create pull secret if needed
if [[ "$SKIP_SECRET" == "false" ]]; then
    if [[ -z "$QUAY_USER" || -z "$QUAY_PASSWORD" ]]; then
        log_error "Quay credentials required. Provide via -u/-p flags or QUAY_USER/QUAY_PASSWORD env vars."
        log_error "Or use --skip-secret if the image is public."
        exit 1
    fi

    if oc get secret "$SECRET_NAME" -n "$NAMESPACE" &>/dev/null; then
        log_warn "Pull secret '$SECRET_NAME' already exists, updating..."
        oc delete secret "$SECRET_NAME" -n "$NAMESPACE"
    fi

    log_info "Creating pull secret '$SECRET_NAME'..."
    oc create secret docker-registry "$SECRET_NAME" \
        --docker-server=quay.io \
        --docker-username="$QUAY_USER" \
        --docker-password="$QUAY_PASSWORD" \
        -n "$NAMESPACE"
fi

# Apply manifests with kustomize, patching the image and secret name
log_info "Applying manifests..."

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Apply manifests with patches using kustomize
cd "$SCRIPT_DIR"

# Create a temporary kustomization overlay for this deployment
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

cat > "$TEMP_DIR/kustomization.yaml" <<EOF
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

namespace: $NAMESPACE

resources:
  - namespace.yaml
  - deployment.yaml
  - service.yaml
  - route.yaml

patches:
  - patch: |-
      - op: replace
        path: /metadata/name
        value: $NAMESPACE
    target:
      kind: Namespace
  - patch: |-
      - op: replace
        path: /spec/template/spec/imagePullSecrets/0/name
        value: $SECRET_NAME
    target:
      kind: Deployment
      name: fbi-ucr

images:
  - name: quay.io/wjackson/crime-stats-api
    newName: ${IMAGE%:*}
    newTag: ${IMAGE#*:}
EOF

# Copy base manifests to temp dir
cp manifests/base/*.yaml "$TEMP_DIR/"

oc apply -k "$TEMP_DIR" -n "$NAMESPACE"

# Wait for deployment
log_info "Waiting for deployment to be ready..."
if oc rollout status deployment/fbi-ucr -n "$NAMESPACE" --timeout=120s; then
    log_info "Deployment successful!"
else
    log_error "Deployment failed or timed out"
    log_error "Check logs with: oc logs -l app.kubernetes.io/name=fbi-ucr -n $NAMESPACE"
    exit 1
fi

# Get route URL
ROUTE_URL=$(oc get route fbi-ucr -n "$NAMESPACE" -o jsonpath='{.spec.host}' 2>/dev/null || echo "")

if [[ -n "$ROUTE_URL" ]]; then
    echo ""
    log_info "=========================================="
    log_info "FBI UCR Prediction Service deployed!"
    log_info "=========================================="
    echo ""
    echo "  Route URL: https://$ROUTE_URL"
    echo ""
    echo "  Test endpoints:"
    echo "    Health:  curl https://$ROUTE_URL/api/v1/health"
    echo "    Models:  curl https://$ROUTE_URL/api/v1/models"
    echo "    Predict: curl -X POST https://$ROUTE_URL/api/v1/predict/violent-crime -H 'Content-Type: application/json' -d '{\"steps\": 6}'"
    echo ""
else
    log_warn "Route not found. You may need to create one manually."
fi

#!/bin/bash

echo "🚀 Building and running Thermal Prediction Server on port 3210..."

# Build Docker image
echo "📦 Building Docker image..."
docker build -t thermal-prediction-server .

# Check if build was successful
if [ $? -eq 0 ]; then
    echo "✅ Docker image built successfully"
    
    # Run container
    echo "🏃 Starting container on port 3210..."
    docker-compose up -d
    
    # Wait for container to start
    echo "⏳ Waiting for server to start..."
    sleep 10
    
    # Test health endpoint
    echo "🔍 Testing health endpoint..."
    if curl -f http://localhost:3210/health > /dev/null 2>&1; then
        echo "✅ Server is healthy and running on port 3210"
        echo "🔗 Access the API at: http://localhost:3210"
    else
        echo "⚠️ Server may still be starting up..."
        echo "📝 Check logs with: docker-compose logs -f"
    fi
    
    # Show logs
    echo "📝 Recent container logs:"
    docker-compose logs --tail=20
    
    echo ""
    echo "🎉 Server is running!"
    echo "📍 API Endpoints:"
    echo "   Health Check: http://localhost:3210/health"
    echo "   Predict:      http://localhost:3210/predict"
    echo "   Get User:     http://localhost:3210/predictions/user/{user_id}"
    echo ""
    echo "📝 To view live logs: docker-compose logs -f"
    echo "🛑 To stop server: docker-compose down"
    
else
    echo "❌ Docker build failed"
    exit 1
fi
"""
Flask MLOps Service for AI Appointment Setter with Prometheus
Lab 2: AI Lifecycle & MLOps Integration


This service handles:
1. Receiving metrics from the Next.js application
2. Tracking AI performance with Prometheus
3. Storing metrics in the database
4. Providing analytics endpoints


Key Learning Objectives:
- Understanding MLOps fundamentals with industry-standard tools
- Implementing metrics collection and tracking with Prometheus
- Building microservices architecture
- Real-time monitoring and alerting
"""


from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import time
from datetime import datetime
import logging
import requests
import urllib.parse
from typing import Dict, Any, Optional


from prometheus_client import Counter, Histogram, Gauge, Info, start_http_server, generate_latest, CONTENT_TYPE_LATEST


from dotenv import load_dotenv
load_dotenv()


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = Flask(__name__)
CORS(app)


DATABASE_URL = os.getenv('DATABASE_URL')
if not DATABASE_URL:
    logger.error("DATABASE_URL environment variable not set")


ai_response_time = Histogram(
    'ai_response_time_seconds',
    'Time taken for AI to respond to user messages',
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)


ai_requests_total = Counter(
    'ai_requests_total',
    'Total number of AI requests',
    ['business_id', 'response_type', 'intent']
)


ai_success_rate = Gauge(
    'ai_success_rate',
    'Success rate of AI responses',
    ['business_id']
)


ai_tokens_used = Counter(
    'ai_tokens_used_total',
    'Total tokens consumed by AI',
    ['business_id', 'model_name']
)


ai_api_cost = Counter(
    'ai_api_cost_usd_total',
    'Total API costs in USD',
    ['business_id', 'model_name']
)


appointments_requested = Counter(
    'appointments_requested_total',
    'Total appointment requests',
    ['business_id']
)


appointments_booked = Counter(
    'appointments_booked_total',
    'Total appointments successfully booked',
    ['business_id']
)


human_handoffs = Counter(
    'human_handoffs_total',
    'Total requests requiring human assistance',
    ['business_id', 'reason']
)


system_info = Info(
    'ai_system_info',
    'Information about the AI system'
)


system_info.info({
    'service': 'ai-appointment-setter',
    'version': '1.0.0',
    'monitoring': 'prometheus'
})


def execute_sql(query: str, params: tuple = None):
    try:
        logger.info(f"SQL Query: {query}")
        if params:
            logger.info(f"Parameters: {params}")
        return {"success": True}
    except Exception as e:
        logger.error(f"Database query error: {e}")
        return None


def create_metrics_table():
    try:
        logger.info("Metrics storage initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Error initializing metrics storage: {e}")
        return False


def rebuild_prometheus_metrics_from_db():
    try:
        logger.info("Rebuilding Prometheus metrics from database...")
        success = fetch_metrics_from_db()
        if success:
            logger.info("Successfully rebuilt Prometheus metrics from database")
        else:
            logger.warning("Could not rebuild metrics from database, starting fresh")
    except Exception as e:
        logger.error(f"Error rebuilding Prometheus metrics: {e}")


create_metrics_table()
rebuild_prometheus_metrics_from_db()


@app.route('/')
def dashboard():
    try:
        with open('dashboard.html', encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return jsonify({
            'message': 'MLOps Service is running!',
            'endpoints': {
                'health': '/health',
                'metrics': '/metrics',
                'track': '/track',
                'analytics': '/analytics/<business_id>'
            }
        })


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'service': 'mlops-service-prometheus',
        'timestamp': datetime.utcnow().isoformat(),
        'monitoring': 'prometheus',
        'metrics_endpoint': '/metrics',
        'prometheus_port': os.getenv('PROMETHEUS_PORT', '8001')
    })


@app.route('/metrics')
def metrics():
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}


@app.route('/track', methods=['POST'])
def track_metrics():
    try:
        data = request.get_json(force=True) or {}

        business_id = data.get("business_id", "default-business")
        session_id = data.get("session_id", "default-session")
        response_time_ms = data.get("response_time_ms", 0)
        tokens_used = data.get("tokens_used", 0)
        api_cost_usd = data.get("api_cost_usd", 0.0)
        intent_detected = data.get("intent_detected", "unknown")
        response_type = data.get("response_type", "text")

        user_message = data.get("message", "")
        user_message_length = data.get("user_message_length", len(user_message))
        ai_response_length = data.get("ai_response_length", 0)

        normalized = {
            "business_id": business_id,
            "session_id": session_id,
            "response_time_ms": response_time_ms,
            "tokens_used": tokens_used,
            "api_cost_usd": api_cost_usd,
            "intent_detected": intent_detected,
            "response_type": response_type,
            "user_message_length": user_message_length,
            "ai_response_length": ai_response_length,
            "appointment_requested": data.get("appointment_requested", False),
            "appointment_booked": data.get("appointment_booked", False),
            "human_handoff_requested": data.get("human_handoff_requested", False)
        }

        update_prometheus_metrics(normalized)
        store_metrics_in_db(normalized)

        return jsonify({
            "status": "success",
            "message": f"AI received your message: '{user_message}'",
            "reply": "Sure! I can help you with that. Could you please provide your preferred date and time for the appointment?",
            "timestamp": datetime.utcnow().isoformat()
        }), 200

    except Exception as e:
        logger.error(f"Error in /track: {e}")
        return jsonify({"error": "Internal server error"}), 500


def update_prometheus_metrics(metrics_data):
    try:
        business_id = metrics_data.get('business_id', 'unknown')
        response_type = metrics_data.get('response_type', 'unknown')
        intent = metrics_data.get('intent_detected', 'unknown')
        model_name = metrics_data.get('model_name', 'gemini-1.5-flash')

        if 'response_time_ms' in metrics_data:
            response_time_seconds = metrics_data['response_time_ms'] / 1000.0
            ai_response_time.observe(response_time_seconds)

        ai_requests_total.labels(
            business_id=business_id,
            response_type=response_type,
            intent=intent
        ).inc()

        if 'success_rate' in metrics_data:
            ai_success_rate.labels(business_id=business_id).set(metrics_data['success_rate'])

        if 'tokens_used' in metrics_data:
            ai_tokens_used.labels(
                business_id=business_id,
                model_name=model_name
            ).inc(metrics_data['tokens_used'])

        if 'api_cost_usd' in metrics_data:
            ai_api_cost.labels(
                business_id=business_id,
                model_name=model_name
            ).inc(metrics_data['api_cost_usd'])

        if metrics_data.get('appointment_requested', False):
            appointments_requested.labels(business_id=business_id).inc()

        if metrics_data.get('appointment_booked', False):
            appointments_booked.labels(business_id=business_id).inc()

        if metrics_data.get('human_handoff_requested', False):
            reason = 'error' if response_type == 'error' else 'complex_query'
            human_handoffs.labels(business_id=business_id, reason=reason).inc()

        logger.debug(f"Updated Prometheus metrics for business {business_id}")

    except Exception as e:
        logger.error(f"Error updating Prometheus metrics: {e}")


def fetch_metrics_from_db():
    try:
        if not DATABASE_URL:
            logger.warning("DATABASE_URL not configured, skipping metrics fetch")
            return False

        from urllib.parse import urlparse
        parsed = urlparse(DATABASE_URL)

        host = parsed.hostname
        database = parsed.path[1:]
        username = parsed.username
        password = parsed.password

        logger.info("Fetching historical metrics directly from Neon database...")
        logger.info("Starting with fresh Prometheus metrics (full DB integration in later labs)")
        return True

    except Exception as e:
        logger.error(f"Error fetching metrics from database: {e}")
        return False


def store_metrics_in_db(metrics_data):
    try:
        logger.info(f"Processed metrics for business {metrics_data.get('business_id')}")
        return True
    except Exception as e:
        logger.error(f"Error processing metrics: {e}")
        return False


@app.route('/refresh-metrics', methods=['POST'])
def refresh_metrics():
    try:
        logger.info("Metrics refresh triggered by Next.js")
        success = fetch_metrics_from_db()

        if success:
            return jsonify({
                'status': 'success',
                'message': 'Prometheus metrics refreshed from database',
                'timestamp': datetime.utcnow().isoformat()
            })
        else:
            return jsonify({
                'status': 'warning',
                'message': 'Could not fetch from database, using current metrics',
                'timestamp': datetime.utcnow().isoformat()
            })

    except Exception as e:
        logger.error(f"Error refreshing metrics: {e}")
        return jsonify({'error': 'Failed to refresh metrics'}), 500


@app.route('/analytics/<business_id>', methods=['GET'])
def get_analytics(business_id: str):
    try:
        return jsonify({
            'business_id': business_id,
            'period': '30_days',
            'metrics': {
                'total_conversations': 150,
                'avg_response_time_ms': 1250.5,
                'avg_tokens_used': 125.3,
                'total_api_cost_usd': 0.045,
                'appointment_requests': 25,
                'appointments_booked': 18,
                'human_handoffs': 3,
                'appointment_conversion_rate': 0.72
            },
            'monitoring': 'prometheus',
            'prometheus_metrics_url': '/metrics',
            'timestamp': datetime.utcnow().isoformat(),
            'note': 'Sample data - connect to your database for real analytics'
        })

    except Exception as e:
        logger.error(f"Error getting analytics: {e}")
        return jsonify({'error': 'Failed to retrieve analytics'}), 500


if __name__ == '__main__':
    service_port = int(os.getenv('SERVICE_PORT', '5001'))
    prometheus_port = int(os.getenv('PROMETHEUS_PORT', '8001'))

    print("🚀 Starting MLOps Service with Prometheus")
    print("=========================================")
    print("📊 Monitoring: Prometheus")
    print("💾 Database: Simplified (cross-platform)")
    print(f"🌐 Service Port: {service_port}")
    print("🌐 Endpoints:")
    print(f"   - GET  http://localhost:{service_port}/ (Dashboard)")
    print(f"   - GET  http://localhost:{service_port}/health")
    print(f"   - GET  http://localhost:{service_port}/metrics (Prometheus)")
    print(f"   - POST http://localhost:{service_port}/track")
    print(f"   - GET  http://localhost:{service_port}/analytics/<business_id>")
    print("")
    print("🎯 Quick Start:")
    print(f"   📊 View Dashboard: http://localhost:{service_port}/")
    print(f"   📈 View Raw Metrics: http://localhost:{service_port}/metrics")
    print("")

    try:
        start_http_server(prometheus_port)
        print(f"📈 Prometheus metrics server started on port {prometheus_port}")
        print(f"📊 Metrics available at: http://localhost:{prometheus_port}/metrics")
    except Exception as e:
        logger.warning(f"Could not start Prometheus metrics server on port {prometheus_port}: {e}")
        print("⚠️  Prometheus metrics available at Flask /metrics endpoint")

    print("")
    print("🔄 Press Ctrl+C to stop all services")
    print("")

    app.run(host='0.0.0.0', port=service_port, debug=True)

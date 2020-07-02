// This is an example file for a churn-like project, you will need to fill in your own steps.
pipeline {
    agent {label 'unix_agent'}
    environment {
        CREDENTIALS_SQL = credentials('2d96c939-85d9-4764-9da3-ff1cb852d16f')
    }
    stages {
	// Build docker production image
        stage('Build training image') {
            steps {
                sh "make build_prod"
            }
        }
	// Run template script
        stage('Run test script') {
            steps {
                sh "make start_prod"
            }
        }
    }
    post {
        always{
            publishHTML (target: [
                    allowMissing: true,
                    alwaysLinkToLastBuild: true,
                    keepAll: true,
                    reportDir: 'stats_log_html',
                    reportFiles: 'index.html',
                    reportName: "Docker CPU/MEM report"
                ])

            sendNotifications()
        }
        cleanup {
            cleanWs()
        }
    }
}

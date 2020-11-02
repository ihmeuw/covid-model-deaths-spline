def conda_dir = "~/.miniconda_deaths-spline"
def env_name="deaths-spline-$BUILD_NUMBER"

def cloneRepoToBuild() {
  sh "echo \$(hostname)"
  sh "pwd"
  sh "echo 'mkdir $BUILD_NUMBER'"
  sh "mkdir $BUILD_NUMBER"
  sh "echo 'Downloading source code...'"
  sh "git clone https://github.com/ihmeuw/covid-model-deaths-spline $BUILD_NUMBER/spline/"
  sh "echo 'Source code downloaded'"
}

pipeline {
  //The Jenkinsfile of ssh://git@stash.ihme.washington.edu:7999/scic/covid-snapshot-etl-orchestration.git

  agent { label 'qlogin' }

  stages{
      stage ('Notify job start'){
         steps{
           emailext body: 'Another email will be send when the job finishes.\n\nMeanwhile, you can view the progress here:\n\n    $BUILD_URL',
                          to: "${EMAIL_TO}",
                          subject: 'Build started in Jenkins: $PROJECT_NAME - #$BUILD_NUMBER'
         }
      }

      stage ('Cleaning'){
        // Don't run when we want to keep results for debugging
        steps{
          node('qlogin') {
            sh "rm -rf * || true"
            }
        }
      }


      stage ('Download source code') {
        steps{
          node('qlogin') {
            cloneRepoToBuild()
            }
           }
      }

      stage ('Generate running file') {
        //We will use the setup_env.sh plus the extra running command
        steps{
          node('qlogin') {
            sh "echo \"#!/bin/bash\" > $BUILD_NUMBER/spline/jenkins_run.sh"
            sh "echo \"export SGE_ENV=prod-el7 \" >> $BUILD_NUMBER/spline/jenkins_run.sh"
            sh "echo \"export SGE_CELL=ihme\" >> $BUILD_NUMBER/spline/jenkins_run.sh"
            sh "echo \"export SGE_ROOT=/opt/sge\" >> $BUILD_NUMBER/spline/jenkins_run.sh"
            sh "echo \"export SGE_CLUSTER_NAME=cluster\" >> $BUILD_NUMBER/spline/jenkins_run.sh"
            sh "echo \"export LC_ALL=en_US.utf-8\" >> $BUILD_NUMBER/spline/jenkins_run.sh"
            sh "echo \"export LANG=en_US.utf-8 \" >> $BUILD_NUMBER/spline/jenkins_run.sh"
            sh "echo \"export PATH=\$PATH:${conda_dir}/bin:/opt/sge/bin/lx-amd64 \" >> $BUILD_NUMBER/spline/jenkins_run.sh"
            sh "echo \"cd $WORKSPACE/$BUILD_NUMBER/spline\" >> $BUILD_NUMBER/spline/jenkins_run.sh"
            sh "dos2unix $BUILD_NUMBER/spline/setup_env.sh"
            script {
              cmd = """tail -n +2 $BUILD_NUMBER/spline/setup_env.sh | sed 's/conda activate/source activate/g' >> $BUILD_NUMBER/spline/jenkins_run.sh"""
              sh "${cmd}"
            }
            sh "echo \"run_deaths -vvv\" >> $BUILD_NUMBER/spline/jenkins_run.sh"
            sh "chmod +x $BUILD_NUMBER/spline/jenkins_run.sh"
            sh "cat $BUILD_NUMBER/spline/jenkins_run.sh"
          }
        }
      }

      stage ('Install miniconda') {
        steps{
          node('qlogin'){
                script{
                   res = sh(script: "test -d ${conda_dir} && echo \"1\" || echo \"0\" ", returnStdout: true).trim()
                   if (res=='1') {
                          sh "echo \"miniconda already installed at $conda_dir\""
                   }else {
                          sh "wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
                          sh "bash Miniconda3-latest-Linux-x86_64.sh -b -p $conda_dir"
                   }
                }
              }
            }
      }

      stage ('Run death') {
        steps{
          node('qlogin'){
            script{
                ssh_cmd = "$WORKSPACE/$BUILD_NUMBER/spline/jenkins_run.sh"
                sshagent(['svccovidci-privatekey']) {
                            sh "ssh -o StrictHostKeyChecking=no svccovidci@int-uge-archive-p012.cluster.ihme.washington.edu \"$ssh_cmd\""
                         }
                 }
          }
        }
      }

     }

  post {
       // Currently only email notification is available on COVID Jenkins. If we want to do slack, we will have to
       // coordinate with INFRA to set it up first. It may request server reboot.
       success {
                emailext body: 'Check console output to view the results:\n\n    $BUILD_URL',
                          to: "${EMAIL_TO}",
                          subject: 'Build succeeded in Jenkins: $PROJECT_NAME - #$BUILD_NUMBER'
        }
       failure {
                 emailext body: 'Check console output to view the results:\n\n    $BUILD_URL',
                          to: "${EMAIL_TO}",
                          subject: 'Build failed in Jenkins: $PROJECT_NAME - #$BUILD_NUMBER'

            }
    }
}
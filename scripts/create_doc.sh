#!/bin/bash
if [ "$TRAVIS_BRANCH" == "gui" ] && [ "$MINICONDA_OS" == "linux" ] && [ "$TRAVIS_PYTHON_VERSION" == "2.7" ]; then
  echo 'Create docs!'
  sphinx-build docs docs/_build/html
  eval "$(ssh-agent -s)"; touch docs/key; chmod 0600 docs/key
  openssl aes-256-cbc -K $encrypted_ba29598036fd_key -iv $encrypted_ba29598036fd_iv -in docs/key.enc -out docs/key -d && ssh-add docs/key
  git config --global user.email "builds@travis-ci.com"
  git config --global user.name "Travis CI"
  git remote set-url --push origin "git@github.com:$TRAVIS_REPO_SLUG"
  export ${!TRAVIS*}
  sphinx-versioning push -r development -w master -w development -b docs gh-pages .
else
  echo 'Did not create docs'
  echo $TRAVIS_BRANCH
  echo $MINICONDA_OS
  echo $TRAVIS_PYTHON_VERSION
fi
exit 0
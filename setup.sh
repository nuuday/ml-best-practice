#!/bin/bash
green='\e[1;32m'
red='\e[31m'
yellow='\e[1;33m'
white='\e[0m'

name=${PWD##*/}

rm -rf .git
git init

printf "\n$green\nCreating repo in BitBucket...$white\n"
url=ssh://git@10.74.139.97:7999/hkrp/$name.git

curl_output=$(curl -s -H "Content-Type: application/json" --user $USER --data '{"name": "'"$name"'"}' https://bitbucket.tdc.dk/rest/api/1.0/projects/HKRP/repos)
if echo $curl_output | grep -q '"errors"'
then
    printf "${red}${curl_output}${white}"
    exit 1
fi
printf "${green}Successfully created repo :)$white\n"

read -r -p "Do you wish to install ai-utils [y/N]? " response

if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]
then
    git submodule add ssh://git@10.74.139.97:7999/hkrp/ai_utils.git
fi

printf "" > README.md

printf "${green}Pushing content to BitBucket${white}"
git remote add origin $url
git add -A
git checkout setup.sh
git commit -m "Initialized project through awesome setup script."
git push --set-upstream origin master

printf "${green}
────────────────────░███░
───────────────────░█░░░█░
──────────────────░█░░░░░█░
─────────────────░█░░░░░█░
──────────░░░───░█░░░░░░█░
─────────░███░──░█░░░░░█░
───────░██░░░██░█░░░░░█░
──────░█░░█░░░░██░░░░░█░
────░██░░█░░░░░░█░░░░█░
───░█░░░█░░░░░░░██░░░█░
──░█░░░░█░░░░░░░░█░░░█░
──░█░░░░░█░░░░░░░░█░░░█░
──░█░░█░░░█░░░░░░░░█░░█░
─░█░░░█░░░░██░░░░░░█░░█░
─░█░░░░█░░░░░██░░░█░░░█░
─░█░█░░░█░░░░░░███░░░░█░
░█░░░█░░░██░░░░░█░░░░░█░
░█░░░░█░░░░█████░░░░░█░
░█░░░░░█░░░░░░░█░░░░░█░
░█░█░░░░██░░░░█░░░░░█░
─░█░█░░░░░████░░░░██░
─░█░░█░░░░░░░█░░██░█░
──░█░░██░░░██░░█░░░█░
───░██░░███░░██░█░░█░
────░██░░░███░░░█░░░█░
──────░███░░░░░░█░░░█░
──────░█░░░░░░░░█░░░█░
──────░█░░░░░░░░░░░░█░
──────░█░░░░░░░░░░░░░█░
──────░█░░░░░░░░░░░░░█░

"

printf "${green}Finished initializing project, this file will now selfdestruct. Bye!${white}\n"

rm -- "$0"
for file in *; do
    # Check if the filename contains uppercase letters
    if [[ $file =~ [A-Z] ]]; then
        # Rename the file to its lowercase version
        mv "$file" "$(echo $file | tr '[:upper:]' '[:lower:]')"
    fi
done

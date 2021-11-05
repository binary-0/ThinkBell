function getTeacher(file) {



    let xhr = new XMLHttpRequest();
    xhr.open('GET', "http://43ea-61-82-78-227.ngrok.io/teacher", true);
    xhr.onload = function () {
        if (this.status === 200) {
            let objects = JSON.parse(this.response);
            console.log(objects);

        }
        else {
            console.error(xhr);
        }
    };
    xhr.send(null);
}
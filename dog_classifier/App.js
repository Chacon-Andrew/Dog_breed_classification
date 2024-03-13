import { StatusBar } from 'expo-status-bar';
import { StyleSheet, Text, View, Button, TouchableOpacity } from 'react-native';
import { useRef, useState } from 'react';
import { useEffect } from 'react';
import { Camera, CameraType } from 'expo-camera'
import { Image } from 'expo-image';
import * as FileSystem from 'expo-file-system'
import { Buffer } from "buffer";

export default function App() {
  const [type, setType] = useState(CameraType.back);
  const [permission, requestPermission] = Camera.useCameraPermissions();
  const [image, setImage] = useState(null)
  const [guess, setGuess] = useState("")
  const cameraref = useRef(null)

  const getGuess = async () => {
    const base64 = await FileSystem.readAsStringAsync(image, { encoding: 'base64'})
    let my_bytes = Buffer.from(base64, "base64")
    const r = await fetch(image)
    const blob = await r.blob()
    var reader = new FileReader()
    const req = {
      img : reader.result
    }
    var headers = {
      'Accept': 'application/json',
      'Content-Type': 'application.json'
    }
    const response = async () => {
      return await fetch('http://683a-2603-7081-1601-a4bd-f134-f8af-c995-8ee8.ngrok-free.app/guess', {
        method: "POST",
        headers: headers,
        body: JSON.stringify(req)
        }).catch(function(error){console.log(error)})
    }
    response().then(resp => resp.json()).then(data => {console.log(data)})
    console.log(guess)
  }

  useEffect(() => {
    if(!permission) requestPermission();
    if(image){
      getGuess()
    }
  }, [image])

  const takePicture = async () => {
    if(cameraref){
      const data = await cameraref.current.takePictureAsync();
      setImage(data.uri)
    }
  }

  return (
    <View style={styles.container}>
      {
        permission != null ? 
        <Camera style={styles.camera} type={type} ref={cameraref} onPress={takePicture}>
          <View style={{position: 'absolute', bottom: 0, marginBottom: 20}}>
            <TouchableOpacity onPress={takePicture} style={styles.button}>
              <Image source={require('./assets/icons8-camera.svg')} style={{width: 100, height: 100}} contentFit='contain'/>
            </TouchableOpacity>
          </View>
        </Camera>
        : <Text>This Application requires the camera to be used</Text>
      }
      <Text>{guess}</Text>
      <StatusBar style="auto" />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
    alignItems: 'center',
    justifyContent: 'center',
  },
  camera: {
    height: '80%',
    width: '100%',
    alignItems: 'center'
  },
  button: {
    height: '100%',
    width: '100%',
  }
});

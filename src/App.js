import { useEffect, useRef, useState } from "react"
import "./App.css"
import image from "./assets/images/image_7.jpg"

import * as bodySegmentation from "@tensorflow-models/body-segmentation"
import * as tf from "@tensorflow/tfjs-core"
import "@tensorflow/tfjs-converter"
// Register WebGL backend.
import "@tensorflow/tfjs-backend-webgl"

const getLabels = (imageData) => {
  var data = new Uint32Array(imageData.data.length / 4)

  for (let i = 0; i < data.length; i++) {
    data[i] = imageData.data[i * 4]
  }

  return data
}

const getPixelsIndex = (label, data) => {
  var index = []

  for (let i = 0; i < data.length; i++) {
    if (data[i] === label) {
      index.push(i)
    }
  }

  return index
}

const getPixelsCoordinate = (indexs, canvasWidth) => {
  var coors = []

  for (let i = 0; i < indexs.length; i++) {
    var y = Math.floor(indexs[i] / canvasWidth)
    var x = indexs[i] % canvasWidth

    coors.push({ x: x, y: y })
  }

  return coors
}

const getPixelsData = async (label, segmentation) => {
  var canvas = document.createElement("canvas")
  var img = document.getElementById("image")
  var context = canvas.getContext("2d")

  canvas.height = img.height
  canvas.width = img.width

  context.drawImage(img, 0, 0)

  const data = await segmentation[0].mask.toImageData()
  var labels = getLabels(data)
  var index = getPixelsIndex(label, labels)
  var pixelCoor = getPixelsCoordinate(index, canvas.width)

  var pixelsData = []

  for (let i = 0; i < pixelCoor.length; i++) {
    var pixelData = context.getImageData(
      pixelCoor[i].x,
      pixelCoor[i].y,
      1,
      1
    ).data

    var red = pixelData[0]
    var green = pixelData[1]
    var blue = pixelData[2]
    var alpha = pixelData[3]

    pixelsData.push({
      coor: { x: pixelCoor[i].x, y: pixelCoor[i].y },
      r: red,
      g: green,
      b: blue,
      a: alpha,
    })
  }

  return pixelsData
}

const getXMax = (pixelsCoor) => {
  return pixelsCoor.reduce(
    (max, pixel) => Math.max(max, pixel.x),
    pixelsCoor[0].x
  )
}

const getXMin = (pixelsCoor) => {
  return pixelsCoor.reduce(
    (min, pixel) => Math.min(min, pixel.x),
    pixelsCoor[0].x
  )
}

const getYMax = (pixelsCoor) => {
  return pixelsCoor.reduce(
    (max, pixel) => Math.max(max, pixel.y),
    pixelsCoor[0].y
  )
}

const getYMin = (pixelsCoor) => {
  return pixelsCoor.reduce(
    (min, pixel) => Math.min(min, pixel.y),
    pixelsCoor[0].y
  )
}

const getCanvasWidth = (pixelsCoor) => {
  const xMax = getXMax(pixelsCoor)
  const xMin = getXMin(pixelsCoor)

  return xMax - xMin
}

const getCanvasHeight = (pixelsCoor) => {
  const yMax = getYMax(pixelsCoor)
  const yMin = getYMin(pixelsCoor)

  return yMax - yMin
}

function App() {
  const [segmenter, setSegmenter] = useState()
  const [img, setImg] = useState()
  const [canvas, setCanvas] = useState()
  const pixelsData = useRef(null)
  const segmentationRef = useRef(null)

  useEffect(() => {
    const prepare = async () => {
      await tf.ready()

      const model = bodySegmentation.SupportedModels.BodyPix
      const segmenterConfig = {
        architecture: "ResNet50",
        outputStride: 16,
        multiplier: 1.0,
        quantBytes: 4,
      }
      const segmenter = await bodySegmentation.createSegmenter(
        model,
        segmenterConfig
      )

      setSegmenter(segmenter)
      setImg(document.getElementById("image"))
      setCanvas(document.getElementById("canvas"))
    }

    prepare()
  }, [])

  useEffect(() => {
    const detect = async (label) => {
      const segmentationConfig = {
        multiSegmentation: false,
        segmentBodyParts: true,
        segmentationThreshold: 0.8,
      }

      if (typeof img !== "undefined") {
        const segmentation = await segmenter.segmentPeople(
          img,
          segmentationConfig
        )

        segmentationRef.current = segmentation

        const coloredPartImage = await bodySegmentation.toColoredMask(
          segmentation,
          bodySegmentation.bodyPixMaskValueToRainbowColor,
          { r: 255, g: 255, b: 255, a: 255 }
        )

        const opacity = 0.7
        const flipHorizontal = false
        const maskBlurAmount = 0
        bodySegmentation.drawMask(
          canvas,
          img,
          coloredPartImage,
          opacity,
          maskBlurAmount,
          flipHorizontal
        )

        pixelsData.current = await getPixelsData(label, segmentationRef.current)
        // console.log(pixelsData.current)
      }
    }

    const drawBodyPart = async (label) => {
      if (img !== undefined) {
        await detect(label)

        const segmentation = segmentationRef.current
        const imageData = await segmentation[0].mask.toImageData()
        const indexs = getPixelsIndex(label, getLabels(imageData))
        const pixelsCoor = getPixelsCoordinate(indexs, canvas.width)
        const pixelsData = await getPixelsData(label, segmentation)
        console.log(pixelsData)

        const height = getCanvasHeight(pixelsCoor)
        const width = getCanvasWidth(pixelsCoor)
        // console.log(width, height)

        const newCanvas = document.getElementById("newCanvas")
        const ctx = newCanvas.getContext("2d")

        newCanvas.width = width
        newCanvas.height = height

        ctx.fillStyle = "red"
        ctx.fillRect(0, 0, newCanvas.width, newCanvas.height)

        const yMin = getYMin(pixelsCoor)
        // const yMax = getYMax(pixelsCoor)
        const xMin = getXMin(pixelsCoor)
        // const xMax = getXMax(pixelsCoor)

        for (let i = 0; i < pixelsData.length; i++) {
            var x = pixelsData[i].coor.x - xMin
            var y = pixelsData[i].coor.y - yMin
            var r = pixelsData[i].r, g = pixelsData[i].g, b = pixelsData[i].b

            ctx.fillStyle = `rgb(${r} ${g} ${b})`
            ctx.fillRect(x, y, 1, 1)
        }

        // for (let y = yMin; y < yMax; y++) {
        //   for (let x = xMin; x < xMax; x++) {
        //     ctx.fillStyle = `rgb(${pixelsData})`
        //   }
        // }
      }
    }

    drawBodyPart(0)
  }, [segmenter])

  //   const getLabelByPixels = (data) => {
  //     var labelByPixels = Array()

  //     for (let i = 0; i < 24; i++) {
  //       labelByPixels.push({ label: i, count: countMaskValue(i, data) })
  //     }

  //     return labelByPixels
  //   }

  //   const countMaskValue = (maskValue, data) => {
  //     var cnt = 0

  //     for (let i = 0; i < data.length; i++) {
  //       if (data[i] === maskValue) {
  //         cnt++
  //       }
  //     }

  //     return cnt
  //   }

  return (
    <div style={{ flexDirection: "row" }}>
      <img
        id="image"
        src={image}
        alt="Khong co gi"
        style={{
          //   position: "absolute",
          textAlign: "center",
          zIndex: 9,
        }}
      ></img>
      <canvas
        id="canvas"
        style={{
          //   position: "absolute",
          textAlign: "center",
          zIndex: 9,
        }}
      ></canvas>
      <canvas id="newCanvas"></canvas>
    </div>
  )
}

export default App

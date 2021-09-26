using System.Collections;
using System.Collections.Generic;
using UnityEngine;

using ILGPU;
using ILGPU.Runtime;
using System;

public class ILGPU_Test : MonoBehaviour
{
    // Start is called before the first frame update
    void Start()
    {
        Context context = Context.Create(builder => builder.AllAccelerators());

        foreach (Device device in context)
        {
            Debug.Log(device);
        }

        context.Dispose();
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}

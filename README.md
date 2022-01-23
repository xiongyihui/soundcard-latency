Soundcard Latency
=================


A tool to check the latency of a sound card.


## Requirements
+ python3
+ numpy
+ python-sounddevice
+ matplotlib

## Get started
+   Measure the default input and output devices.

    ```
    python3 latency.py
    ```

+   Measure two specified devices

    1.  list all devices

        ```
        python3 -m sounddevice
        ```

    2.  use the indexes to specify two devices

        ```
        python3 latency 1 2
        ```


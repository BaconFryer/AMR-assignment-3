�
    ��fs  �                   �.   � d Z dZ G d� d�  �        Zd� ZdS )F�   c                   �   � e Zd Zd� Zd� ZdS )�PIDControllerc                 �L   � || _         || _        || _        d| _        d| _        d S )N�    )�kp�ki�kd�
prev_error�integral)�selfr   r   r	   s       �*/home/kris/MEGA/work/AMR/CW3/controller.py�__init__zPIDController.__init__   s)   � ������������������    c                 ��   � | xj         ||z  z  c_         || j        z
  |z  }|| _        t          d|� d| j         � d|� ��  �         | j        |z  | j        | j         z  z   | j        |z  z   S )NzError: z, Integral: z, Derivative: )r   r
   �printr   r   r	   )r   �error�dt�
derivatives       r   �updatezPIDController.update   s}   � �������#����d�o�-��3�
�����T��T�T�4�=�T�T�
�T�T�U�U�U��w�����4�=�!8�8�4�7�Z�;O�O�Or   N)�__name__�
__module__�__qualname__r   r   � r   r   r   r      s7   � � � � � �� � �P� P� P� P� Pr   r   c                 �  � | d         }| d         }| d         }| d         }| d         }| d         }|d         |z
  }	|d         |z
  }
d}d}d	}d}d}d	}d}d
}d
}d}d}d}t          |||�  �        }t          |||�  �        }t          |||�  �        }t          |||�  �        }t          |||�  �        }t          |||�  �        }d}|�                    |	|�  �        }|�                    |
|�  �        }||z
  } ||z
  }!|"|z
  }#|�                    | |�  �        }$|�                    |!|�  �        }%|�                    | |�  �        }"|�                    |#|�  �        }&t          t          |"d�  �        d�  �        }"t          t          |&d�  �        d�  �        }&|"|z
  }#|&|z
  }'|�                    |'|�  �        }(||%z
  |(z   })||%z
  |(z
  }*t          t          |)d�  �        d�  �        })t          t          |*d�  �        d�  �        }*|)|*f}+|+S )Nr   �   �   �   �   �   g      �?g�������?g�������?g{�G�z�?g       @g�������?g��Q��?g�������?g������ɿg      �?g      �g        )r   r   �max�min),�state�
target_posr   �x_pos�y_pos�x_vel�y_vel�att�ang_vel�	x_pos_err�	y_pos_err�kp_pos�ki_pos�kd_pos�kp_vel�ki_vel�kd_vel�kp_att�ki_att�kd_att�
kp_ang_vel�
ki_ang_vel�
kd_ang_vel�	pid_x_pos�	pid_y_pos�	pid_x_vel�	pid_y_vel�pid_att�pid_ang_vel�base�desired_x_vel�desired_y_vel�	x_vel_err�	y_vel_err�desired_att�att_err�
x_throttle�
y_throttle�desired_ang_vel�ang_vel_err�ang_throttle�u_1�u_2�actions,                                               r   �
controllerrM      sf  � � �!�H�E��!�H�E��!�H�E��!�H�E�
��(�C��A�h�G��1���%�I��1���%�I� �F��F��F� �F��F��F� �F��F��F� �J��J��J��f�f�f�5�5�I��f�f�f�5�5�I��f�f�f�5�5�I��f�f�f�5�5�I��F�F�F�3�3�G��
�J�
�C�C�K��D� �$�$�Y��3�3�M��$�$�Y��3�3�M� ��%�I���%�I��C��G� �!�!�)�R�0�0�J��!�!�)�R�0�0�J� �"�"�9�b�1�1�K��n�n�W�b�1�1�O� �c�+�s�+�+�T�2�2�K��#�o�s�3�3�T�:�:�O� �C��G�!�G�+�K� �%�%�k�2�6�6�L� ��
�l�
*�C�
��
�l�
*�C� �c�#�s�m�m�S�
!�
!�C�
�c�#�s�m�m�S�
!�
!�C��3�Z�F��Mr   N)�wind_active�group_numberr   rM   r   r   r   �<module>rP      s]   ������P� P� P� P� P� P� P� P�"S� S� S� S� Sr   
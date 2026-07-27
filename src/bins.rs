//! Maps the model's 7 fine-grained classes into the 3 waste bins the app
//! reports. Edit `bin_for_class` below to match your district's sorting rules —
//! that's the one spot non-ML changes to sorting logic live.

/// The model's class ids, in training order. Do not reorder — these indices
/// come straight out of the ONNX `output0` class column.
pub const CLASS_NAMES: [&str; 7] = [
    "Chips",        // 0
    "Compost",      // 1
    "Napkin",       // 2
    "Plastic-Cups", // 3
    "Plastic-Wrap", // 4
    "Recycle",      // 5
    "Utensils",     // 6
];

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Bin {
    Compost,
    Recycle,
    Trash,
}

impl Bin {
    pub fn label(self) -> &'static str {
        match self {
            Bin::Compost => "COMPOST",
            Bin::Recycle => "RECYCLE",
            Bin::Trash => "TRASH",
        }
    }

    /// RGB overlay color for boxes/labels of this bin.
    pub fn color(self) -> [u8; 3] {
        match self {
            Bin::Compost => [46, 160, 67],  // green
            Bin::Recycle => [31, 111, 235], // blue
            Bin::Trash => [180, 60, 60],    // red
        }
    }
}

/// Class-id -> bin. This is the district-specific sorting policy. Best-guess
/// defaults below; confirm with whoever owns the cafeteria waste contract.
pub fn bin_for_class(class_id: usize) -> Bin {
    match class_id {
        1 => Bin::Compost, // Compost
        2 => Bin::Compost, // Napkin (unlined paper -> compostable)
        3 => Bin::Recycle, // Plastic-Cups
        5 => Bin::Recycle, // Recycle
        0 => Bin::Trash,   // Chips (foil-lined bag)
        4 => Bin::Trash,   // Plastic-Wrap (film -> not curbside recyclable)
        6 => Bin::Trash,   // Utensils
        _ => Bin::Trash,
    }
}

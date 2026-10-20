//! Maps the model's 37 fine-grained classes into the waste bins the app
//! reports. Edit `bin_for_name` below to match your district's sorting rules —
//! that's the one spot non-ML changes to sorting logic live.

/// The model's class ids, in training order. Do not reorder — these indices
/// are the class-score rows of the ONNX `output0` tensor.
pub const CLASS_NAMES: [&str; 37] = [
    "Bread",           // 0
    "Burrito",         // 1
    "Can",             // 2
    "Cardboard-Tray",  // 3
    "Carton",          // 4
    "Cheese",          // 5
    "Chicken",         // 6
    "Chip-Bag",        // 7
    "Condiments",      // 8
    "Corndog",         // 9
    "Fries",           // 10
    "Fruit",           // 11
    "Generic-Wrapper", // 12
    "Gravy",           // 13
    "Hamburger",       // 14
    "Human",           // 15
    "Lasagna",         // 16
    "Macaroni",        // 17
    "Mashed Potatoes", // 18
    "Meat",            // 19
    "Napkin",          // 20
    "Pizza",           // 21
    "Plastic-Tray",    // 22
    "Plastic-Wrapper", // 23
    "Quesadilla",      // 24
    "Rice",            // 25
    "Salad",           // 26
    "Sauce",           // 27
    "Snack-Cup",       // 28
    "Spaghetti",       // 29
    "Styrofoam-Cup",   // 30
    "Styrofoam-Tray",  // 31
    "Tortilla",        // 32
    "Utensil",         // 33
    "Vegetables",      // 34
    "Waffle",          // 35
    "Water-Bottle",    // 36
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bin {
    Compost,
    Recycle,
    Trash,
    /// People in frame: drawn, but never counted toward a bin.
    Human,
    /// Unknown class id — flagged rather than silently binned as Trash.
    NeedsReview,
}

impl Bin {
    pub fn label(self) -> &'static str {
        match self {
            Bin::Compost => "COMPOST",
            Bin::Recycle => "RECYCLE",
            Bin::Trash => "TRASH",
            Bin::Human => "HUMAN",
            Bin::NeedsReview => "NEEDS REVIEW",
        }
    }

    /// RGB overlay color for boxes/labels of this bin.
    pub fn color(self) -> [u8; 3] {
        match self {
            Bin::Compost => [0, 255, 0],       // green
            Bin::Recycle => [100, 100, 255],   // light blue
            Bin::Trash => [255, 0, 0],         // red
            Bin::Human => [0, 0, 0],           // black
            Bin::NeedsReview => [255, 190, 0], // orange
        }
    }
}

pub fn class_name(class_id: usize) -> &'static str {
    CLASS_NAMES.get(class_id).copied().unwrap_or("?")
}

/// Class-id -> bin. Out-of-range ids are `NeedsReview`.
pub fn bin_for_class(class_id: usize) -> Bin {
    CLASS_NAMES
        .get(class_id)
        .map_or(Bin::NeedsReview, |name| bin_for_name(name))
}

/// Class name -> bin. This is the district-specific sorting policy.
fn bin_for_name(name: &str) -> Bin {
    match name {
        "Plastic-Tray" | "Can" | "Snack-Cup" | "Water-Bottle" | "Chip-Bag" | "Carton" => {
            Bin::Recycle
        }
        // Cardboard-Tray could reasonably go to Compost instead — move it to
        // that arm if your district composts soiled cardboard.
        "Cardboard-Tray" => Bin::Recycle,
        "Styrofoam-Tray" | "Utensil" | "Plastic-Wrapper" | "Napkin" | "Generic-Wrapper"
        | "Condiments" | "Styrofoam-Cup" | "Gravy" | "Sauce" => Bin::Trash,
        "Hamburger" | "Fries" | "Chicken" | "Waffle" | "Salad" | "Fruit" | "Quesadilla"
        | "Bread" | "Vegetables" | "Burrito" | "Meat" | "Cheese" | "Spaghetti" | "Pizza"
        | "Corndog" | "Mashed Potatoes" | "Lasagna" | "Macaroni" | "Tortilla" | "Rice" => {
            Bin::Compost
        }
        "Human" => Bin::Human,
        _ => Bin::NeedsReview,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_class_maps_to_expected_bin() {
        use Bin::*;
        let expected = [
            Compost, // 0 Bread
            Compost, // 1 Burrito
            Recycle, // 2 Can
            Recycle, // 3 Cardboard-Tray
            Recycle, // 4 Carton
            Compost, // 5 Cheese
            Compost, // 6 Chicken
            Recycle, // 7 Chip-Bag
            Trash,   // 8 Condiments
            Compost, // 9 Corndog
            Compost, // 10 Fries
            Compost, // 11 Fruit
            Trash,   // 12 Generic-Wrapper
            Trash,   // 13 Gravy
            Compost, // 14 Hamburger
            Human,   // 15 Human
            Compost, // 16 Lasagna
            Compost, // 17 Macaroni
            Compost, // 18 Mashed Potatoes
            Compost, // 19 Meat
            Trash,   // 20 Napkin
            Compost, // 21 Pizza
            Recycle, // 22 Plastic-Tray
            Trash,   // 23 Plastic-Wrapper
            Compost, // 24 Quesadilla
            Compost, // 25 Rice
            Compost, // 26 Salad
            Trash,   // 27 Sauce
            Recycle, // 28 Snack-Cup
            Compost, // 29 Spaghetti
            Trash,   // 30 Styrofoam-Cup
            Trash,   // 31 Styrofoam-Tray
            Compost, // 32 Tortilla
            Trash,   // 33 Utensil
            Compost, // 34 Vegetables
            Compost, // 35 Waffle
            Recycle, // 36 Water-Bottle
        ];
        assert_eq!(expected.len(), CLASS_NAMES.len());
        for (id, want) in expected.iter().enumerate() {
            assert_eq!(bin_for_class(id), *want, "class {id} {}", CLASS_NAMES[id]);
        }
    }

    #[test]
    fn out_of_range_needs_review() {
        assert_eq!(bin_for_class(37), Bin::NeedsReview);
        assert_eq!(bin_for_class(usize::MAX), Bin::NeedsReview);
        assert_eq!(class_name(37), "?");
    }
}
